import numpy as np
from sklearn.base import BaseEstimator, ClusterMixin
from tools.config import N_CLUSTERS

FuzzyCMeansParamsGrid = {
    "n_clusters": [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "fuzzyness": [1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5]
}

# Implementation of the Generalized Suppressed Fuzzy C-means algorithm
class FuzzyCMeans(ClusterMixin, BaseEstimator):
    def __init__(self, n_clusters: int, fuzzyness: float, suppression_rule='theta', suppression_param=0.5):
        self.n_clusters = n_clusters
        self.fuzzyness = fuzzyness
        self.suppression_rule = suppression_rule  # 'theta', 'rho', 'beta', 'kappa', 'tau', 'sigma', 'xi'
        self.suppression_param = suppression_param

    def fit(self, X):
        X = X.to_numpy()
        n_samples, n_features = X.shape
        
        # Initialize cluster prototypes (randomly for this example)
        self.cluster_prototypes_ = X[np.random.choice(n_samples, self.n_clusters, replace=False)]

        # Initialize fuzzy membership matrix
        U = self._initialize_membership(X)

        # Main loop (alternating optimization)
        while True:
            previous_prototypes = np.copy(self.cluster_prototypes_)

            # 4. Compute distances
            distances = self._compute_distances(X)

            # 5. Update fuzzy membership matrix
            U = self._update_membership(distances)

            # 6. Apply suppression (context-sensitive)
            U = self._apply_suppression(U, distances)

            # 7. Update cluster prototypes
            self.cluster_prototypes_ = self._update_prototypes(X, U)

            # 8. Check for convergence
            diff = np.linalg.norm(self.cluster_prototypes_ - previous_prototypes)
            
            if diff < 1e-4:
                break


        self.is_fitted_ = True
        self.clusters_ = np.argmax(U, axis=1)
        self.centroids_ = self.cluster_prototypes_
        
        return self
    
    def fit_predict(self, data):
        """
        Fit the model and return cluster labels.
        """
        self.fit(data)
        return self.clusters_

    def _initialize_membership(self, X):
        U = np.random.rand(len(X), self.n_clusters)
        U = U / np.sum(U, axis=1, keepdims=True)  # Normalize to satisfy probabilistic constraint
        return U

    def _compute_distances(self, X):
        distances = np.zeros((len(X), self.n_clusters))
        for i in range(self.n_clusters):
            distances[:, i] = np.linalg.norm(X - self.cluster_prototypes_[i], axis=1)
        return distances

    def _update_membership(self, distances):
        m = self.fuzzyness
        U = np.power(distances, -2 / (m - 1-1e-6))
        U = np.where(np.isinf(U), 1.0, U) # handles possible inf values
        U = U / np.sum(U, axis=1, keepdims=True)  # Normalize

        return U

    def _apply_suppression(self, U, distances):
        # print('apply suppression')
        suppressed_U = np.zeros_like(U)
        winner_indices = np.argmax(U, axis=1)

        for k in range(len(U)):
            w = winner_indices[k]  # Winner cluster index
            alpha_k = self._compute_suppression_rate(U[k, w])

            suppressed_U[k, :] = alpha_k * U[k, :]
            suppressed_U[k, w] = 1 - alpha_k + alpha_k * U[k, w]

        return suppressed_U

    def _compute_suppression_rate(self, u_w):
        m = self.fuzzyness
        rule = self.suppression_rule
        param = self.suppression_param

        if rule == 'theta':
            return 1 / (1 - u_w + u_w * (1 - param) ** (2 / (1 - m)))
        elif rule == 'rho':
            return 1 / (1 - u_w + param ** (2 / (1 - m)) * u_w ** ((3 - m) / (1 - m)))
        elif rule == 'beta':
            return 1 / (1 + u_w * (u_w ** (2 * param / (1 - m) / (1 - param)) - 1))
        elif rule == 'kappa':
            return 1 / (1 - u_w + u_w * (0.5 - (2 * param - 1) / 2 * np.sin(np.pi * u_w)) ** (2 / (1 - m)))
        elif rule == 'tau':
            return (1 - param) / (1 + u_w * param)
        elif rule == 'sigma':
            return (1 - u_w ** param) / (1 - u_w)
        elif rule == 'xi':
            return (1 - (np.sin(np.pi * u_w / 2)) ** param) / (1 - u_w)
        else:
            raise ValueError("Invalid suppression rule")

    def _update_prototypes(self, X, U):
        m = self.fuzzyness
        U_m = np.power(U, m)
        new_prototypes = np.dot(U_m.T, X) / np.sum(U_m, axis=0, keepdims=True).T
        return new_prototypes