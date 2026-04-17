import time

import torch
from xrfm.rfm_src.recursive_feature_machine import RFM, ClassificationConverter, matrix_power, tenumerate, with_env_var

class resrfm(RFM):

    # TODO: to prevent OOM issue due to memory fragmentation but doesn't actually set the environment variable!
    @with_env_var("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True") 
    def fit(self, train_data, val_data=None, iters=None, method='lstsq', reg=None, center_grads=False,
            verbose=False, M_batch_size=None, ep_epochs=None, return_best_params=True, bs=None, 
            return_Ms=False, lr_scale=1, total_points_to_sample=None, solver=None, 
            tuning_metric=None, prefit_eigenpro=True, early_stop_rfm=True, early_stop_multiplier=1.1, 
            callback=None, **kwargs):
        """
        :param train_data: tuple of (X, y)
        :param val_data: tuple of (X, y)
        :param iters: number of iterations to run
        :param method: 'lstsq' or 'eigenpro'
        :param reg: Regularization coefficient (higher is more regularization).
        :param classification: if True, the model will tune for (and report) accuracy, else just MSE loss
        :param verbose: if True, print progress
        :param M_batch_size: batch size over samples for AGOP computation
        :param return_best_params: if True, return the best parameters
        :param bs: batch size for eigenpro
        :param return_Ms: if True, return the Mahalanobis matrix at each iteration
        :param lr_scale: learning rate scale for EigenPro
        :param total_points_to_sample: number of points to sample for AGOP computation
        :param solver: 'solve' or 'cholesky' or 'lu', used in LSTSQ computation
        :param prefit_eigenpro: if True, prefit EigenPro with a subset of <= max_lstsq_size samples
        """

        # Initialize parameters
        self._initialize_fit_parameters(iters, method, reg, verbose, M_batch_size, total_points_to_sample,
                                       ep_epochs, tuning_metric, early_stop_rfm, early_stop_multiplier,
                                       center_grads, prefit_eigenpro, solver, **kwargs)
        
        

        # Validate and prepare data
        X_train, y_train, X_val, y_val = self.validate_data(train_data, val_data)
        n, d = X_train.shape
        print("="*70)
        print(f"Fitting RFM with ntrain: {n}, d: {d}, and nval: {X_val.shape[0]}")
        print("="*70)


        if self.class_converter is None:
            self.class_converter = ClassificationConverter(mode='zero_one', n_classes=max(2, y_train.shape[1]))


        self.adapt_params_to_data(n, d)
        
        # Initialize tracking variables
        metrics, Ms = [], []
        best_alphas, best_M, best_sqrtM = None, None, None
        best_metric = float('inf') if self.should_minimize else float('-inf')
        best_iter = None
        early_stopped = False
        best_bandwidth = self.kernel_obj.bandwidth+0

        start_time = time.time()

        # Main training loop
        for i in range(self.iters):
            # check time limit
            if i > 0 and self.time_limit_s is not None and (i+1)/i*(time.time()-start_time) > self.time_limit_s:
                break  # would expect to exceed the time limit, so stop


            if callback is not None:
                callback(iteration=i)

            start = time.time()
            self.fit_predictor(X_train, y_train, X_val=X_val, y_val=y_val, 
                               bs=bs, lr_scale=lr_scale, **kwargs)
                        
            # Compute validation metrics
            val_metrics = self._compute_validation_metrics(X_train, y_train, X_val, y_val, iteration_num=i, M_batch_size=M_batch_size, **kwargs)

            # Update best parameters if needed
            if return_best_params:
                best_metric, best_alphas, best_M, best_sqrtM, best_iter, best_bandwidth = self.update_best_params(
                    best_metric, best_alphas, best_M, best_sqrtM, best_iter, best_bandwidth, 
                    val_metrics[self.tuning_metric], i)
             
            # Check for early stopping
            if self.early_stop_rfm:
                val_metric = val_metrics[self.tuning_metric]
                if self._should_early_stop(val_metric, best_metric):
                    print(f"Early stopping at iteration {i}")
                    if not return_best_params:
                        self.fit_M(X_train, y_train.shape[-1], M_batch_size=M_batch_size, **kwargs)
                    early_stopped = True
                    break

            # Fit M matrix and cleanup
            self.fit_M(X_train, y_train.shape[-1], M_batch_size=M_batch_size, **kwargs)
            del self.weights
            
            if return_Ms:
                Ms.append(self.tensor_copy(self.M))
                metrics.append(val_metrics[self.tuning_metric])

            print(f"Time taken for round {i}: {time.time() - start} seconds")

        if callback is not None:
            callback(iteration=self.iters)

        # Handle final iteration if no early stopping occurred
        if not early_stopped:
            self.fit_predictor(X_train, y_train, X_val=X_val, y_val=y_val, bs=bs, **kwargs)        
            final_val_metrics = self._compute_validation_metrics(X_train, y_train, X_val, y_val, is_final=True, **kwargs)

            if return_best_params:
                best_metric, best_alphas, best_M, best_sqrtM, best_iter, best_bandwidth = self.update_best_params(
                    best_metric, best_alphas, best_M, best_sqrtM, best_iter, best_bandwidth, 
                    final_val_metrics[self.tuning_metric], iters)
                
        # Restore best parameters
        if return_best_params:
            self.M = None if best_M is None else best_M.to(self.device)
            self.sqrtM = None if best_sqrtM is None else best_sqrtM.to(self.device)
            self.weights = best_alphas.to(self.device) # type: ignore
            self.kernel_obj.bandwidth = best_bandwidth

        self.best_iter = best_iter

        if self.verbose:
            print(f"{self.best_iter=}")

        if kwargs.get('get_agop_best_model', False):
            # fit AGOP of best model
            self.agop_best_model = self.fit_M(X_train, y_train, M_batch_size=M_batch_size, inplace=False, **kwargs)

        self.residual_weighted_agop_best_model = self.fit_residual_weighted_agop(X_train, y_train, M_batch_size=M_batch_size, inplace=False, **kwargs)
        return Ms if return_Ms else None
    
    def fit_residual_weighted_agop(self, samples, targets, M_batch_size=None, inplace=True, **kwargs):
        """
        Fit the Mahalanobis matrix M using AGOP.
        
        Parameters
        ----------
        samples : torch.Tensor
            Input samples of shape (n_samples, n_features)
        targets : torch.Tensor
            Target values of shape (n_samples, n_targets)
        M_batch_size : int, optional
            Batch size for AGOP computation. If None, computed automatically
            based on available memory.
        inplace : bool, default=True
            Whether to update self.M and self.sqrtM in place. If False, returns
            the computed M matrix without modifying the object.
        **kwargs : dict
            Additional arguments (unused, for compatibility)
            
        Returns
        -------
        torch.Tensor or None
            If inplace=False, returns the computed M matrix. Otherwise returns None.
            
        Notes
        -----
        - This method computes a residual-weighted AGOP, where the contributions 
        - of samples to the AGOP matrix are weighted by their residuals (the difference 
        - between predicted and true targets).
        """
        
        n, d = samples.shape
        M = torch.zeros_like(self.M) if self.M is not None else (
            torch.zeros(d, dtype=samples.dtype, device=self.device) 
            if self.diag else torch.zeros(d, d, dtype=samples.dtype, device=self.device))
        

        if M_batch_size is None: 
            BYTES_PER_SCALAR = samples.element_size()
            M_batch_size = self._compute_optimal_M_batch(n, targets.shape[-1], d, scalar_size=BYTES_PER_SCALAR)
        
        if targets.ndim == 1:
            targets = targets.unsqueeze(-1)
        targets = targets.to(self.device)

        residual_weights = self._compute_residual_weights(samples, targets)

        batches = torch.arange(n).split(M_batch_size)

        num_batches = 1 + self.total_points_to_sample//M_batch_size
        batches = batches[:num_batches]
        if self.verbose:
            print(f'Sampling AGOP on maximum of {num_batches*M_batch_size} total points')

        if self.verbose:
            for i, bids in tenumerate(batches):
                M.add_(self.update_M_with_residuals(samples[bids], residual_weights[bids]))
        else:
            for bids in batches:
                M.add_(self.update_M_with_residuals(samples[bids], residual_weights[bids]))
        
        scaled_M = M / (M.max() + 1e-30)
        if self.use_sqrtM:
            sqrtM = matrix_power(scaled_M, self.agop_power)
        else:
            sqrtM = None
        
        if inplace:
            self.M = scaled_M
            self.sqrtM = sqrtM
        else:
            return scaled_M

    def _compute_residual_weights(self, samples, targets):
        preds = self.predict(samples)

        residuals = (preds.to(targets.dtype) - targets).mean(dim=1) # type: ignore
        return residuals
    
    def update_M_with_residuals(self, samples, residual_weights):
        """
        Update the Mahalanobis matrix M using AGOP on a batch of samples.
        
        Parameters
        ----------
        samples : torch.Tensor
            Input samples of shape (n_samples, n_features)
        residual_weights : torch.Tensor
            Residual-based sample weights of shape (n_samples,)

        Returns
        -------
        torch.Tensor
            AGOP matrix of shape (n_features, n_features) or (n_features,) if diagonal
        """
        samples = samples.to(self.device)
        self.centers = self.centers.to(self.device)
        
        if self.M is None:
            if self.diag:
                self.M = torch.ones(samples.shape[-1], device=samples.device, dtype=samples.dtype)
            else:
                self.M = torch.eye(samples.shape[-1], device=samples.device, dtype=samples.dtype)

        if self.use_sqrtM and self.sqrtM is None:
            if self.diag:
                self.sqrtM = torch.ones(samples.shape[-1], device=samples.device, dtype=samples.dtype)
            else:
                self.sqrtM = torch.eye(samples.shape[-1], device=samples.device, dtype=samples.dtype)

        f_grads = self.kernel_obj.get_function_grads(
            x=self.centers,
            z=samples,
            coefs=self.weights.t(),
            mat=self.sqrtM if self.use_sqrtM else self.M,
        )

        residual_weights = residual_weights.to(device=samples.device, dtype=samples.dtype)
        residual_weights = self.apply_phi(residual_weights)
        residual_weights = residual_weights.sqrt() # sqrt because we will apply the weights to the gradients, and the outer product will multiply the weights
        f_grads = f_grads * residual_weights[None,:, None]

        f_grads = f_grads.reshape(-1, f_grads.shape[-1])
        if self.center_grads:
            f_grads = f_grads - f_grads.mean(dim=0, keepdim=True)

        if self.diag:
            return f_grads.square().sum(dim=-2)
        return f_grads.transpose(-1, -2) @ f_grads
    
    def apply_phi(self, X):
        return X.square()