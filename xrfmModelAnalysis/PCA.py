import pandas as pd
from sklearn.decomposition import PCA


def calculatePCA(X: pd.DataFrame, n_components: int) -> tuple[pd.DataFrame, PCA]:
    """Calculate PCA on the given data and return the transformed data and the PCA object"""
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)
    X_pca_df = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(n_components)])
    return X_pca_df, pca