# IML Theory Summary

## T0: Overview of Machine Learning

### Key Concepts:

1. **Machine Learning Types:**
    
    - **Supervised Learning:** Learn a mapping from inputs to outputs using labeled data. Examples include regression and classification.
    - **Unsupervised Learning:** Discover hidden patterns or structure in unlabeled data. Examples include clustering and dimensionality reduction.
    - **Reinforcement Learning:** Learn to make sequences of decisions by receiving rewards or penalties.
2. **Applications:**
    
    - Spam filtering (classification)
    - Customer segmentation (clustering)
    - Game-playing AI (reinforcement learning)
3. **Common Algorithms:**
    
    - Linear regression, logistic regression, decision trees, SVMs, and neural networks.
4. **Pipeline:**
    
    - Problem definition -> Data collection and preprocessing -> Model training -> Model evaluation -> Deployment.

---

## T1: Basic Concepts of Supervised Learning

### Key Concepts:

1. **Model Components:**
    
    - **Input Features (X):** Variables used to make predictions.
    - **Target Variable (Y):** The outcome or label to predict.
2. **Learning Process:**
    
    - Minimize the error between predictions and actual targets (e.g., using Mean Squared Error for regression).
3. **Bias-Variance Tradeoff:**
    
    - **Bias:** Error due to overly simplistic models.
    - **Variance:** Error due to overly complex models fitting noise.
    - Goal: Achieve a balance to reduce total error.
4. **Overfitting vs. Underfitting:**
    
    - **Overfitting:** Model performs well on training data but poorly on new data.
    - **Underfitting:** Model fails to capture the underlying trend in the data.
5. **Evaluation Metrics:**
    
    - **Classification:** Accuracy, precision, recall, F1 score.
    - **Regression:** Mean Squared Error (MSE), R-squared.

### Example:

- **Spam Detection:** Use email text as input features (e.g., word frequencies) and spam/ham labels as target variables.

---

## T2: Classification Algorithms

### Key Algorithms:

1. **Logistic Regression:**
    
    - Predicts probabilities for binary outcomes using the sigmoid function.
2. **Support Vector Machines (SVM):**
    
    - Finds a hyperplane that maximizes the margin between classes.
    - Kernel tricks enable SVMs to work in high-dimensional feature spaces.
3. **Decision Trees:**
    
    - Splits data based on feature values to maximize information gain or minimize impurity (e.g., Gini index, entropy).
4. **K-Nearest Neighbors (KNN):**
    
    - Classifies based on the majority class of the nearest neighbors in the feature space.
5. **Naïve Bayes:**
    
    - Probabilistic classifier assuming feature independence.

### Example:

- **Iris Dataset Classification:** Predict the species of an iris flower based on features like petal length and width.

---

## T3: Regression Algorithms

### Key Algorithms:

1. **Linear Regression:**
    
    - Models a linear relationship between input features and the target variable.
    - Equation: Y = b0 + b1X1 + … + bnXn
2. **Polynomial Regression:**
    
    - Extends linear regression to model nonlinear relationships.
3. **Ridge and Lasso Regression:**
    
    - Regularization techniques to prevent overfitting.
    - Ridge adds L2 penalty; Lasso adds L1 penalty.
4. **Gradient Boosting:**
    
    - Builds an ensemble of weak learners (e.g., decision trees) to minimize the loss function iteratively.
5. **Random Forests:**
    
    - An ensemble of decision trees trained on random subsets of data and features.

### Example:

- **House Price Prediction:** Use features like number of rooms, location, and area to predict the price of a house.

---

## T4: Model Evaluation and Validation

### Key Concepts:

1. **Train-Test Split:**
    
    - Split data into training and testing sets to evaluate performance on unseen data.
2. **Cross-Validation:**
    
    - Divide data into k subsets (folds); train on k-1 folds and validate on the remaining fold. Repeat for all folds.
3. **Evaluation Metrics:**
    
    - **Classification:** Confusion matrix, precision, recall, F1 score, ROC curve.
    - **Regression:** R-squared, MSE, MAE.
4. **Regularization:**
    
    - Adds penalties to model complexity to avoid overfitting.
5. **Grid Search and Hyperparameter Tuning:**
    
    - Systematically search for the best hyperparameters using cross-validation.

### Example:

- **ROC Curve in Cancer Diagnosis:** Visualize trade-offs between true positive rate and false positive rate at different thresholds.

---
## T5: Ensemble Learning and Model Aggregation

### Key Concepts:

1. **Ensemble Learning:**
    
    - Combines multiple models to improve overall performance.
    - Types:
        - Bagging: Builds independent models and combines predictions (e.g., Random Forests).
        - Boosting: Builds models sequentially, correcting errors from prior models (e.g., AdaBoost, Gradient Boosting).
        - Stacking: Combines predictions from different types of models using a meta-model.
2. **Bagging Methods:**
    
    - **Random Forests:**
        - Uses random subsets of data and features to build decision trees.
        - Reduces overfitting by averaging predictions.
3. **Boosting Methods:**
    
    - **AdaBoost:**
        - Assigns weights to data points; misclassified points get higher weights in subsequent models.
    - **Gradient Boosting:**
        - Minimizes loss function by iteratively building models to correct residual errors.

### Example:

- **Spam Detection:** Combining weak classifiers (e.g., decision trees) using boosting to improve spam email detection accuracy.

---

## T6: Dimensionality Reduction

### Key Concepts:

1. **Why Dimensionality Reduction?**
    
    - Simplifies datasets by reducing the number of features while retaining meaningful information.
    - Addresses issues like the curse of dimensionality and enhances visualization.
2. **Principal Component Analysis (PCA):**
    
    - Projects data onto a lower-dimensional space while preserving variance.
    - Steps:
        - Compute the covariance matrix.
        - Calculate eigenvectors and eigenvalues.
        - Select principal components (eigenvectors with highest eigenvalues).
    - **Example:** Visualizing a 3D dataset in 2D using PCA.
3. **Independent Component Analysis (ICA):**
    
    - Focuses on finding statistically independent components in data.
    - Often used in signal processing (e.g., separating mixed audio signals).
4. **t-SNE and UMAP:**
    
    - Nonlinear dimensionality reduction techniques.
    - Preserve neighborhood relationships for visualization.
5. **Linear Discriminant Analysis (LDA):**
    
    - Supervised dimensionality reduction that maximizes class separability.

### Example:

- **Facial Recognition:** Using PCA to reduce high-dimensional image data for efficient processing.

---

## T7: Introduction to Unsupervised Learning and Clustering

### Key Concepts:

1. **Unsupervised Learning:**
    
    - Finds hidden structure in unlabeled data.
    - Goals: Summarization and understanding.
2. **Clustering Algorithms:**
    
    - **K-Means Clustering:**
        - Divides data into k clusters by minimizing intra-cluster variance.
        - Iterative algorithm: Assign data points to clusters, update centroids.
    - **Hierarchical Clustering:**
        - Builds a tree-like dendrogram by either merging or splitting clusters.
        - Types: Agglomerative (bottom-up) and divisive (top-down).
    - **DBSCAN:**
        - Groups points based on density.
        - Handles noise and clusters of arbitrary shapes.
3. **Evaluation Metrics:**
    
    - Silhouette score, Davies-Bouldin index, and elbow method.
4. **Applications:**
    
    - Bioinformatics, image segmentation, market research.

### Example:

- **Market Segmentation:** Grouping customers into clusters based on purchasing behavior.

---

## T8: Advanced Clustering Techniques and Soft Clustering

### Key Concepts:

1. **Soft Clustering:**
    
    - Assigns probabilities to data points for belonging to multiple clusters.
    - Example: Fuzzy C-Means.
2. **Expectation-Maximization (EM):**
    
    - Soft clustering approach using probability distributions.
    - Alternates between expectation (assign probabilities) and maximization (update parameters).
3. **Mixture Models:**
    
    - Represents data as a mixture of underlying probability distributions (e.g., Gaussian Mixture Models).
4. **Clustering Challenges:**
    
    - Choosing the number of clusters.
    - Defining similarity or distance measures.
5. **Real-World Applications:**
    
    - Document clustering, social network analysis, medical imaging.

### Example:

- **Image Segmentation:** Using fuzzy clustering to classify pixels into overlapping regions (e.g., edges, textures).

---
## T9: Factor Analysis and Dimensionality Reduction

### Key Concepts:

1. **Factor Analysis (FA):**
    
    - Reduces observed variables into a smaller number of unobserved latent variables (factors).
    - Models observed variables as linear combinations of latent factors and error terms.
2. **Principal Component Analysis (PCA):**
    
    - Identifies orthogonal axes (principal components) to explain variance.
    - Differences from FA:
        - PCA focuses on variance, while FA focuses on covariance and latent factors.
3. **Independent Component Analysis (ICA):**
    
    - Decomposes multivariate data into statistically independent components.
    - Applications: Signal processing (e.g., separating audio sources).
4. **Eigenvectors and Eigenvalues:**
    
    - Critical to PCA for determining the axes of maximum variance.

### Example:

- **Cocktail Party Problem:** ICA separates overlapping sound signals into independent audio tracks.

---

## T10: Visualization Techniques

### Key Concepts:

1. **Why Visualization?**
    
    - Understand and interpret high-dimensional data in lower dimensions.
2. **Techniques:**
    
    - **Self-Organizing Maps (SOM):**
        - Neural networks that map high-dimensional data onto a 2D or 3D grid while preserving topological relationships.
    - **Multi-Dimensional Scaling (MDS):**
        - Assigns points in low-dimensional space to preserve dissimilarity between data points.
        - Types:
            - Metric MDS (preserves absolute distances).
            - Non-metric MDS (preserves rank-order relationships).
    - **Projection Methods:**
        - PCA, Sammon's projection, and Canonical Correlation Analysis (CCA).
3. **Advantages:**
    
    - Reveals hidden structures in data.
    - Can handle nonlinear relationships (e.g., t-SNE, UMAP).

### Example:

- **Iris Dataset:** Visualizing the relationships between iris species in 2D using MDS or t-SNE.

---

## T11: Recommender Systems

### Key Concepts:

1. **Purpose:**
    
    - Provide personalized recommendations to users based on preferences and behaviors.
2. **Techniques:**
    
    - **Content-Based Filtering:**
        - Recommends items similar to those a user liked previously based on item features.
    - **Collaborative Filtering (CF):**
        - **User-Based:** Finds similar users and recommends their preferred items.
        - **Item-Based:** Finds similar items to those the user interacted with.
    - **Hybrid Methods:** Combines multiple approaches for improved recommendations.
3. **Challenges:**
    
    - Cold-start problem (lack of initial data).
    - Sparsity (limited data overlap among users).
    - Popularity bias (favoring widely-rated items).
4. **Evaluation Metrics:**
    
    - Precision, recall, F1 score, and Mean Absolute Error (MAE).

### Example:

- **Movie Recommendation:** Suggesting films based on similar users' preferences (CF) or similar genres (content-based).

---

## T12: Bias and Fairness in Machine Learning

### Key Concepts:

1. **Bias:**
    
    - Prejudice or imbalance in data, algorithms, or outcomes.
    - Examples:
        - **Data Bias:** Non-representative training data.
        - **Algorithm Bias:** Model assumptions that disadvantage certain groups.
2. **Fairness:**
    
    - Ensures equitable treatment of individuals/groups in ML outcomes.
    - Definitions include:
        - Anti-classification: Excludes protected attributes (e.g., race, gender).
        - Classification Parity: Equal predictive performance across groups.
        - Calibration: Consistent probabilities across groups.
3. **Real-World Examples:**
    
    - COMPAS: Bias in predicting recidivism rates based on race.
    - Amazon Hiring Algorithm: Discrimination against female candidates due to biased training data.
4. **Bias in Recommender Systems:**
    
    - **Selection Bias:** Non-random user behavior affects data.
    - **Exposure Bias:** Popular items overrepresented in recommendations.
    - **Conformity Bias:** Users rate similarly due to group behavior.
5. **Mitigation Techniques:**
    
    - Pre-processing: Adjust training data.
    - In-processing: Modify algorithms to include fairness constraints.
    - Post-processing: Adjust outputs to achieve fairness.

### Example:

- **Gender Shades Project:** Highlighting lower accuracy of facial recognition systems for darker-skinned females.

---
