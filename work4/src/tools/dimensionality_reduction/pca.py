from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.decomposition import PCA as SklearnPCA

PCA_PARAMS_GRID_MAP = {
    "n_components": [2, 3],
    # "svd_solver": ["auto"],  
}

# TODO: implement our own PCA
class PCA(TransformerMixin, BaseEstimator):
    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.pca = SklearnPCA(n_components=self.n_components)

    def fit(self, X, y=None):
        self.pca.fit(X)
        return self

    def transform(self, X):
        return self.pca.transform(X)
    


import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import TransformerMixin, BaseEstimator

class OurPCA(TransformerMixin, BaseEstimator):
    def __init__(self, n_components: int = 2):
        """
        Custom PCA implementation.
        :param n_components: Number of principal components to keep.
        """
        self.n_components = n_components
        self.mean_vector = None
        self.eigenvalues = None
        self.eigenvectors = None
        self.sorted_eigenvalues = None
        self.sorted_eigenvectors = None

    def fit(self, X, y=None):
        """
        Fit the PCA model to the dataset.
        :param X: Input data (n_samples, n_features)
        :param y: Ignored, for compatibility.
        :return: self
        """
        # Compute mean vector
        self.mean_vector = np.mean(X, axis=0)

        # Center the data
        centered_data = X - self.mean_vector

        # Compute covariance matrix
        covariance_matrix = np.cov(centered_data, rowvar=False)

        # Ensure covariance matrix is symmetric
        covariance_matrix = (covariance_matrix + covariance_matrix.T) / 2

        # Perform eigenvalue decomposition
        eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

        # Use only real parts
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)

        # Sort eigenvalues and eigenvectors
        sorted_indices = np.argsort(eigenvalues)[::-1]
        self.sorted_eigenvalues = eigenvalues[sorted_indices]
        self.sorted_eigenvectors = eigenvectors[:, sorted_indices]

        return self

    def transform(self, X):
        """
        Transform the data into the PCA subspace.
        :param X: Input data (n_samples, n_features)
        :return: Transformed data (n_samples, n_components)
        """
        if self.sorted_eigenvectors is None:
            raise ValueError("PCA not fitted. Call `fit` first.")

        # Center the data
        centered_data = X - self.mean_vector

        # Select the top k eigenvectors
        projection_matrix = self.sorted_eigenvectors[:, :self.n_components]

        # Project the data
        return np.dot(centered_data, projection_matrix)

    def inverse_transform(self, X_transformed):
        """
        Reconstruct the data back to the original space.
        :param X_transformed: Transformed data (n_samples, n_components)
        :return: Reconstructed data (n_samples, n_features)
        """
        projection_matrix = self.sorted_eigenvectors[:, :self.n_components]
        return np.dot(X_transformed, projection_matrix.T) + self.mean_vector

    def plot_original_data(self, X, feature_names=None):
        """
        Plot the original data (first two features).
        :param X: Input data (n_samples, n_features)
        :param feature_names: List of feature names for labels.
        """
        plt.figure(figsize=(8, 6))
        plt.scatter(X[:, 0], X[:, 1], c='blue', alpha=0.5, label='Original Data')
        plt.title("Original Data Scatter Plot")
        plt.xlabel(feature_names[0] if feature_names else "Feature 1")
        plt.ylabel(feature_names[1] if feature_names else "Feature 2")
        plt.legend()
        plt.show()

    def plot_transformed_data(self, X_transformed):
        """
        Plot the transformed data in the PCA subspace.
        :param X_transformed: Transformed data (n_samples, n_components)
        """
        plt.figure(figsize=(8, 6))
        plt.scatter(X_transformed[:, 0], X_transformed[:, 1], c='green', alpha=0.5, label='PCA Transformed Data')
        plt.title("PCA Transformed Data Scatter Plot")
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.legend()
        plt.show()

    def plot_reconstructed_data(self, X_original, X_reconstructed):
        """
        Plot the original and reconstructed data.
        :param X_original: Original data (n_samples, n_features)
        :param X_reconstructed: Reconstructed data (n_samples, n_features)
        """
        plt.figure(figsize=(8, 6))
        plt.scatter(X_original[:, 0], X_original[:, 1], c='blue', alpha=0.5, label='Original Data')
        plt.scatter(X_reconstructed[:, 0], X_reconstructed[:, 1], c='red', alpha=0.5, label='Reconstructed Data')
        plt.title("Original vs Reconstructed Data")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.legend()
        plt.show()

# # Helper function to load data from .arff file
# def load_arff(file_path):
#     data, meta = arff.loadarff(file_path)
#     df = pd.DataFrame(data)
#     df_numeric = df.select_dtypes(include=[np.number])  # Ignore non-numeric columns (e.g., class labels)
#     return df_numeric.to_numpy(), df_numeric.columns

# # Example usage
# if __name__ == "__main__":
#     # Step 1: Load dataset
#     file_path = 'your_dataset.arff'  # Replace with your file path
#     data, feature_names = load_arff(file_path)
#     print("Data loaded successfully!")

#     # Initialize PCA object
#     pca = PCA(n_components=2)

#     # Step 2: Plot original data
#     pca.plot_original_data(data, feature_names)

#     # Step 3-6: Fit PCA
#     pca.fit(data)

#     # Step 7: Transform data
#     transformed_data = pca.transform(data)
#     print("Transformed Data Shape:", transformed_data.shape)

#     # Step 8: Plot transformed data
#     pca.plot_transformed_data(transformed_data)

#     # Step 9: Reconstruct data and plot
#     reconstructed_data = pca.inverse_transform(transformed_data)
#     pca.plot_reconstructed_data(data, reconstructed_data)

