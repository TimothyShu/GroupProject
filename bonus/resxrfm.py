import torch
from xrfm import xRFM

from resrfm import resrfm

from xrfm.rfm_src.gpu_utils import memory_scaling_factor

class xRFM_res(xRFM): 
    # to keep the result as close to xRFM as possible, 
    # we will just override the methods that change 
    # best_agop_model which will now apply the residual
    
    def _get_agop_on_subset(self, X, y, subset_size=50_000, time_limit_s=None, max_subset_size_for_split_rfm=60_000):
        """

        This method fits a base RFM model on a subset of the data to compute the AGOP matrix,
        whose eigenvectors are used to generate projection direction for data splitting.

        Parameters
        ----------
        X : torch.Tensor
            Input features of shape (n_samples, n_features)
        y : torch.Tensor
            Target values of shape (n_samples, n_targets)
        subset_size : int, default=50000
            Maximum size of the subset to use for AGOP computation
        max_subset_size_for_split_rfm : int, default=60000
            Maximum size of the subset to use for AGOP computation

        Returns
        -------
        torch.Tensor
            AGOP matrix of shape (n_features, n_features)
        """
        # This is the only change we need to make
        model = resrfm(**self.default_rfm_params['model'], device=self.device, time_limit_s=time_limit_s, # type: ignore
                    **self.extra_rfm_params_)

        base_subset_size = int(subset_size)
        scaled_subset_size = max(int(base_subset_size * memory_scaling_factor(self.device, quadratic=True)), 1)
        subset_size = min(scaled_subset_size, len(X), max_subset_size_for_split_rfm)
        subset_train_size = max(int(subset_size * 0.95), 1)  # 95/5 split, probably won't need the val data.

        subset_indices = torch.randperm(len(X))
        subset_train_indices = subset_indices[:subset_train_size]
        subset_val_indices = subset_indices[subset_train_size:subset_size]

        X_train = X[subset_train_indices]
        y_train = y[subset_train_indices]
        X_val = X[subset_val_indices]
        y_val = y[subset_val_indices]

        print("Getting AGOP on subset")
        print("X_train", X_train.shape, "y_train", y_train.shape, "X_val", X_val.shape, "y_val", y_val.shape)

        model.fit((X_train, y_train), (X_val, y_val), **self.default_rfm_params['fit'])
        

        # use residual weighted agop if specified, this is the only change we need to make to get the residual weighted agop
        residual_weighted_agop = model.residual_weighted_agop_best_model
        if residual_weighted_agop is not None:
            print("Successfully computed residual weighted AGOP")
            return residual_weighted_agop

        # fallback to regular agop if not using residual weighted agop or if there was an issue with fitting the residual weighted agop
        agop = model.agop_best_model

        return agop

