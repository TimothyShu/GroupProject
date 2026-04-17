import numpy as np
import torch
from sklearn.model_selection import train_test_split


def compute_holdout_metrics(model, X, y, metric="accuracy"):
	# Ensure X is a numpy array for torch compatibility
	X_input = X.to_numpy()
	preds = model.predict(X_input)
	# Ensure preds is a torch tensor
	if isinstance(preds, np.ndarray):
		preds = torch.from_numpy(preds)
	# Ensure targets is a torch tensor
	targets = torch.tensor(y.to_numpy(), dtype=torch.float32)

	# print shapes
	print(f"Preds shape: {preds.shape}")
	print(f"Targets shape: {targets.shape}")

	# Squeeze to match shapes
	preds = torch.squeeze(preds)
	targets = torch.squeeze(targets)

	metrics = {}

	# Try to detect classification vs regression
	is_classification = False
	if hasattr(model, 'class_converter_') and hasattr(model.class_converter_, 'numerical_to_labels'):
		# Heuristic: if targets are integer and small number of unique values, treat as classification
		if torch.is_floating_point(targets):
			unique_vals = torch.unique(targets)
			if len(unique_vals) <= 20 and torch.all(unique_vals == unique_vals.long().float()):
				is_classification = True
		else:
			is_classification = True

	if is_classification or metric == "accuracy":
		# Classification
		targets = model.class_converter_.numerical_to_labels(targets)
		targets = torch.squeeze(targets)
		if preds.dtype != targets.dtype:
			preds = preds.to(targets.dtype)
		metrics["accuracy"] = (preds == targets).float().mean().item()
	else:
		# Regression
		if preds.dtype != targets.dtype:
			preds = preds.to(targets.dtype)
		metrics["mse"] = torch.mean((preds - targets) ** 2).item()

	# print 1 element of preds and targets to check if they are in the same format
	print(f"Sample prediction: {preds[:5]}")
	print(f"Sample target: {targets[:5]}")

	return metrics


def print_holdout_metrics(name, metrics):
	print(f"\n{name} holdout test metrics")
	for metric_name, metric_value in metrics.items():
		print(f"  {metric_name} = {metric_value:.6f}")


def compare_model_accuracy(X, y, model1, model2, metric="accuracy"):

	model1_metrics = compute_holdout_metrics(model1, X, y, metric=metric)
	model2_metrics = compute_holdout_metrics(model2, X, y, metric=metric)

	print_holdout_metrics("Model 1", model1_metrics)
	print_holdout_metrics("Model 2", model2_metrics)

	return {
		"xrfm": model1_metrics,
		"resxrfm": model2_metrics,
	}