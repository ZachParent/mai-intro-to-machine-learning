- [ ] Copy files (.gitignore, Makefile, README.md, requirements.txt, setup.cfg)
- [x] setup data directories
  - [x] 0_raw
  - [x] 1_preprocessed
  - [x] 2_clustered
    - <dataset_name>
      - [ ] <model_name>
        - [ ] <config_name>.csv
  - [x] 3_metrics
    - <dataset_name>
      - [ ] <model_name>
        - [ ] <config_name>.csv
- [ ] start a python module
  - [ ] scripts dir
    - [ ] run_preprocessing.py
        - runs for all datasets at once
    - [ ] run_model.py
        - takes in a dataset and a model, run all configs
    - [ ] run_metrics.py
        - runs for all results at once
    - [ ] run_analysis.py
        - runs for all results at once
  - [ ] tools dir
    - [ ] config.py
    - [ ] clustering_methods
        - [ ] optics.py
        - [ ] spectral_clustering.py
        - [ ] kmeans.py
        - [ ] improved_kmeans_A.py
        - [ ] improved_kmeans_B.py
        - [ ] fuzzy_cmeans.py
    - [ ] analysis
        - [ ] tables.py
        - [ ] plots.py
        - [ ] metrics.py
- [ ] create synthetic data
- [ ] fill KMeans with sklearn kmeans
- [ ] generate some results



```python
run_spectral_clustering_params_grid = {
    'n_clusters': [2, 3, 4, 5, 6, 7, 8, 9, 10],
    'affinity': ['nearest_neighbors', 'rbf'],
    'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
}

class SpectralClustering:
    def __init__(self, n_clusters, affinity, gamma):
        self.n_clusters = n_clusters
        self.affinity = affinity
        self.gamma = gamma

    def fit(self, data):
        pass

    def predict(self):
        pass
```

```python
k_means_params_grid = {
    'n_clusters': [2, 3, 4, 5, 6, 7, 8, 9, 10]
}

class KMeans(BaseEstimator):
    def __init__(self, n_clusters, init_clusters=None):
        self.n_clusters = n_clusters
        self.init_clusters = init_clusters

    def fit(self, data):
        pass

    def predict(self):
        pass

class ImprovedKMeansA:
    def __init__(self, n_cluster):
        self.n_clusters = n_clusters

    def fit(self, data):
        kmeans = KMeans(self.n_clusters)
        generate_initial_centroids(data, self.n_clusters)
        kmeans.fit(data)
```

```python
from spectral_clustering import run_spectral_clustering, run_spectral_clustering_params_grid

param_list = run_spectral_clustering_params_grid
GridSearchCV(run_spectral_clustering, param_list)
``` 