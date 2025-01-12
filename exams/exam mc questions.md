# Quiz: T1 to T4

## T1: Basic Concepts of Supervised Learning

### Question 1

Select the correct statement. Which of the following is an example of supervised learning?

Select one:

- a. Clustering documents based on their topics
- b. Predicting the price of a house based on historical data
- c. Segmenting an image to identify objects
- d. Finding patterns in an unlabeled dataset

### Question 2

Which of the following best describes overfitting?

Select one:

- a. A model that performs well on unseen data but poorly on training data
- b. A model that memorizes training data and performs poorly on unseen data
- c. A model that generalizes well across different datasets
- d. A model that uses too few features to make predictions

### Question 3

Which of the following is an example of a regression task?

Select one:

- a. Identifying spam emails
- b. Predicting the temperature for the next week
- c. Categorizing animals in an image
- d. Clustering customers based on purchasing behavior

### Question 4

What is the primary goal of a cost function in supervised learning?

Select one:

- a. To evaluate the accuracy of the model
- b. To calculate the complexity of the dataset
- c. To measure the difference between predicted and actual outputs
- d. To determine the best learning rate for gradient descent

### Question 5

Which of the following metrics is used for regression tasks?

Select one:

- a. Precision
- b. Mean Squared Error (MSE)
- c. F1 score
- d. Confusion matrix

### Question 6

The bias-variance tradeoff refers to:

Select one:

- a. The balance between the model's performance on training and testing data
- b. The tradeoff between model complexity and training time
- c. The compromise between underfitting and overfitting
- d. The balance between true positives and false negatives

### Question 7

Which of the following statements about supervised learning is true?

Select one:

- a. It works with labeled and unlabeled data
- b. It works only with unlabeled data
- c. It requires labeled data for training
- d. It doesn’t use a target variable

### Question 8

In a classification task, which metric combines precision and recall?

Select one:

- a. Accuracy
- b. ROC-AUC
- c. F1 Score
- d. Mean Absolute Error (MAE)

### Question 9

Which of the following techniques is used to handle overfitting?

Select one:

- a. Increasing the number of features
- b. Adding noise to the target variable
- c. Using regularization techniques
- d. Reducing the size of the training set

### Question 10

What is the primary purpose of training and testing splits in supervised learning?

Select one:

- a. To improve training time
- b. To evaluate model performance on unseen data
- c. To reduce computational complexity
- d. To generate more features for the model

---

## Correct Answers for T1

1. b
2. b
3. b
4. c
5. b
6. c
7. c
8. c
9. c
10. b

---

## T2: Classification Algorithms

### Question 11

Which algorithm finds a hyperplane that maximizes the margin between classes?

Select one:

- a. K-Nearest Neighbors
- b. Decision Tree
- c. Support Vector Machine (SVM)
- d. Logistic Regression

### Question 12

In K-Nearest Neighbors (KNN), what is the role of the parameter `k`?

Select one:

- a. Number of features used in training
- b. Size of the dataset
- c. Number of nearest neighbors considered for classification
- d. Number of clusters to divide the data into

### Question 13

Which of the following is a probabilistic classifier?

Select one:

- a. Decision Tree
- b. Naïve Bayes
- c. SVM
- d. KNN

### Question 14

Which of the following techniques is used to prevent overfitting in decision trees?

Select one:

- a. Regularization
- b. Splitting the dataset
- c. Pruning
- d. Increasing the depth of the tree

### Question 15

What type of data is required for logistic regression?

Select one:

- a. Numerical and continuous
- b. Binary or categorical outcomes
- c. Ordinal data
- d. Text data only

### Question 16

What is the role of the sigmoid function in logistic regression?

Select one:

- a. To map the output to probabilities
- b. To reduce dimensionality
- c. To select features
- d. To calculate loss

### Question 17

In a classification task, what is the purpose of a confusion matrix?

Select one:

- a. To measure the distance between data points
- b. To visualize model errors across categories
- c. To reduce the complexity of the dataset
- d. To evaluate the variance in predictions

### Question 18

Which of the following represents the equation for a decision boundary in logistic regression?

Select one:

- a. h(x) = w^T x + b
- b. h(x) = θ_0 + θ_1x
- c. h(x) = sign(w^T x)
- d. h(x) = sigmoid(w^T x)

### Question 19

Which technique improves SVM performance in non-linear separable datasets?

Select one:

- a. Kernel Trick
- b. Bagging
- c. Feature Scaling
- d. Boosting

### Question 20

Which evaluation metric is commonly used for imbalanced classification problems?

Select one:

- a. Accuracy
- b. ROC-AUC
- c. MSE
- d. Mean Absolute Error

---

## Correct Answers for T2

11. c
12. c
13. b
14. c
15. b
16. a
17. b
18. d
19. a
20. b

---

## T3: Regression Algorithms

### Question 21

Which regression technique adds an L1 penalty to the cost function?

Select one:

- a. Ridge Regression
- b. Lasso Regression
- c. Logistic Regression
- d. Polynomial Regression

### Question 22

What is the main purpose of regularization in regression models?

Select one:

- a. To increase training accuracy
- b. To reduce overfitting
- c. To make predictions faster
- d. To improve interpretability of features

### Question 23

Which algorithm fits a linear model by minimizing the squared errors?

Select one:

- a. Ridge Regression
- b. Linear Regression
- c. Decision Trees
- d. Gradient Boosting

### Question 24

What is a key characteristic of polynomial regression?

Select one:

- a. It uses logarithmic transformations
- b. It models non-linear relationships
- c. It applies regularization techniques
- d. It clusters data points

### Question 25

In gradient boosting, how are subsequent models trained?

Select one:

- a. By minimizing errors made by the previous models
- b. By maximizing data variance
- c. By creating independent trees
- d. By focusing on feature selection

### Question 26

Which of the following is true about Random Forests?

Select one:

- a. It uses boosting to combine weak learners
- b. It builds trees in a sequential manner
- c. It creates an ensemble of decision trees using bagging
- d. It requires fewer features to build each tree

### Question 27

Which metric is commonly used to evaluate regression models?

Select one:

- a. Precision
- b. Recall
- c. Mean Absolute Error (MAE)
- d. ROC-AUC

### Question 28

What does the term "multicollinearity" refer to in regression analysis?

Select one:

- a. High correlation between independent variables
- b. A linear relationship between dependent and independent variables
- c. High variance in predictions
- d. Non-linear data patterns

### Question 29

Which of the following is an advantage of ridge regression?

Select one:

- a. Automatically selects features
- b. Reduces the impact of multicollinearity
- c. Performs well on sparse datasets
- d. Uses probabilistic models for predictions

### Question 30

What is the purpose of the cost function in regression?

Select one:

- a. To calculate accuracy
- b. To minimize prediction errors
- c. To determine the number of features
- d. To maximize variance

---

## Correct Answers for T3

21. b
22. b
23. b
24. b
25. a
26. c
27. c
28. a
29. b
30. b

---

## T4: Model Evaluation and Validation

### Question 31

What is the primary purpose of cross-validation?

Select one:

- a. To generate additional data
- b. To evaluate the model’s performance on multiple datasets
- c. To improve the learning rate
- d. To reduce the complexity of the model

### Question 32

What is the "k" in k-fold cross-validation?

Select one:

- a. The number of iterations for gradient descent
- b. The size of the training dataset
- c. The number of folds the dataset is divided into
- d. The number of features used in training

### Question 33

Which of the following methods is used for hyperparameter tuning?

Select one:

- a. Regularization
- b. Grid Search
- c. Cross-validation
- d. Feature Scaling

### Question 34

What is the purpose of a learning curve?

Select one:

- a. To determine the number of hidden layers in a neural network
- b. To visualize training and testing performance as a function of dataset size
- c. To compare different optimization algorithms
- d. To calculate the learning rate for gradient descent

### Question 35

Which of the following statements is true about regularization?

Select one:

- a. L1 regularization tends to produce sparse models
- b. L2 regularization completely eliminates multicollinearity
- c. Regularization is only applied to classification problems
- d. Regularization decreases the number of training samples

### Question 36

Which metric is most appropriate for evaluating a classification model on imbalanced data?

Select one:

- a. Accuracy
- b. F1 Score
- c. Mean Squared Error
- d. R-Squared

### Question 37

Why is it important to separate data into training and validation sets?

Select one:

- a. To reduce computation time
- b. To ensure the model is not overfitting to the training data
- c. To increase the dataset size
- d. To improve feature scaling

### Question 38

Which of the following can cause data leakage during validation?

Select one:

- a. Using cross-validation
- b. Including test data in the training set
- c. Performing feature scaling
- d. Using a high learning rate

### Question 39

When is early stopping used during training?

Select one:

- a. When the model's loss stops improving on the validation set
- b. When the model starts overfitting to the test data
- c. When gradient descent diverges
- d. When the number of epochs exceeds a predefined value

### Question 40

Which metric is typically used to evaluate the performance of a regression model?

Select one:

- a. Precision
- b. Recall
- c. Mean Absolute Error
- d. F1 Score

---

## Correct Answers for T4

31. b
32. c
33. b
34. b
35. a
36. b
37. b
38. b
39. a
40. c

## T5: Ensemble Learning and Model Aggregation

### Question 41

What is the primary purpose of ensemble learning?

Select one:

- a. To reduce the size of the dataset
- b. To combine multiple models to improve performance
- c. To decrease training time
- d. To simplify model complexity

### Question 42

Which ensemble method creates multiple models by training on random subsets of data?

Select one:

- a. Boosting
- b. Stacking
- c. Bagging
- d. Feature Scaling

### Question 43

In Random Forests, how are trees trained?

Select one:

- a. Sequentially, correcting errors from previous trees
- b. Independently, using randomly sampled data and features
- c. By optimizing the maximum margin between classes
- d. By clustering data points before training

### Question 44

Which boosting algorithm adjusts the weights of data points after each iteration?

Select one:

- a. Gradient Boosting
- b. AdaBoost
- c. Random Forests
- d. Bagging

### Question 45

What is a primary difference between bagging and boosting?

Select one:

- a. Bagging focuses on reducing bias, while boosting focuses on reducing variance
- b. Bagging trains models independently, while boosting trains models sequentially
- c. Boosting is only applicable to classification tasks
- d. Bagging uses fewer models than boosting

### Question 46

What is the role of a meta-model in stacking?

Select one:

- a. To aggregate predictions from base models
- b. To prune decision trees
- c. To initialize base models
- d. To perform clustering before training

### Question 47

Which of the following is true about Gradient Boosting Machines (GBM)?

Select one:

- a. They use decision trees as weak learners
- b. They are based on clustering algorithms
- c. They only work on regression tasks
- d. They cannot handle missing values

### Question 48

Which metric is commonly optimized in ensemble models for classification tasks?

Select one:

- a. Mean Absolute Error
- b. F1 Score
- c. R-Squared
- d. Sum of Squared Errors

### Question 49

How does Random Forest handle overfitting?

Select one:

- a. By pruning trees to reduce depth
- b. By using cross-validation during training
- c. By averaging predictions across multiple trees
- d. By adding regularization terms to cost functions

### Question 50

What is a potential drawback of ensemble learning methods?

Select one:

- a. They are computationally expensive
- b. They are less accurate than single models
- c. They cannot handle large datasets
- d. They do not work with categorical features

---

## Correct Answers for T5

41. b
42. c
43. b
44. b
45. b
46. a
47. a
48. b
49. c
50. a

---

## T6: Dimensionality Reduction

### Question 51

What is the primary goal of dimensionality reduction?

Select one:

- a. To increase the number of features in a dataset
- b. To reduce the number of features while preserving meaningful information
- c. To cluster data points into distinct groups
- d. To improve the accuracy of a classification model

### Question 52

Which of the following techniques is a linear dimensionality reduction method?

Select one:

- a. PCA (Principal Component Analysis)
- b. t-SNE (t-distributed Stochastic Neighbor Embedding)
- c. UMAP (Uniform Manifold Approximation and Projection)
- d. SOM (Self-Organizing Maps)

### Question 53

What is a key assumption of PCA?

Select one:

- a. Variables are statistically independent
- b. Data follows a Gaussian distribution
- c. Variance explains important features
- d. Clusters are linearly separable

### Question 54

What does ICA aim to achieve in dimensionality reduction?

Select one:

- a. Minimize variance between features
- b. Find independent components in data
- c. Create a low-dimensional projection for clustering
- d. Identify clusters in categorical data

### Question 55

Which dimensionality reduction technique focuses on preserving local relationships?

Select one:

- a. PCA
- b. LDA (Linear Discriminant Analysis)
- c. t-SNE
- d. Factor Analysis

### Question 56

What is the main output of a PCA transformation?

Select one:

- a. Clusters of similar data points
- b. A set of orthogonal principal components
- c. A list of independent features
- d. A similarity matrix

### Question 57

How does regularization impact dimensionality reduction techniques?

Select one:

- a. It increases the number of features selected
- b. It removes correlated variables from datasets
- c. It prevents overfitting by penalizing complexity
- d. It improves the interpretability of clusters

### Question 58

Which application is best suited for ICA?

Select one:

- a. Image compression
- b. Signal separation
- c. Clustering customers
- d. Visualizing high-dimensional data

### Question 59

Which of the following methods is commonly used for visualizing high-dimensional data?

Select one:

- a. PCA
- b. t-SNE
- c. UMAP
- d. All of the above

### Question 60

What is one drawback of t-SNE compared to PCA?

Select one:

- a. It cannot handle large datasets efficiently
- b. It doesn’t work on non-linear relationships
- c. It requires data to be linearly separable
- d. It outputs non-interpretable principal components

---

## Correct Answers for T6

51. b
52. a
53. c
54. b
55. c
56. b
57. c
58. b
59. d
60. a

---

## T7: Introduction to Unsupervised Learning and Clustering

### Question 61

What is the primary goal of clustering in unsupervised learning?

Select one:

- a. Predict labels for new data points
- b. Group similar data points together
- c. Train models on labeled datasets
- d. Minimize error rates

### Question 62

Which of the following algorithms is a density-based clustering method?

Select one:

- a. K-Means
- b. DBSCAN
- c. Hierarchical Clustering
- d. Mean Shift

### Question 63

What is the role of the "k" parameter in K-Means clustering?

Select one:

- a. Number of clusters
- b. Number of iterations
- c. Number of features
- d. Number of neighbors

### Question 64

Which clustering method can handle non-spherical clusters?

Select one:

- a. K-Means
- b. DBSCAN
- c. Agglomerative Clustering
- d. Hierarchical Clustering

### Question 65

What is a limitation of K-Means clustering?

Select one:

- a. It cannot handle categorical data
- b. It is computationally expensive
- c. It assumes clusters are spherical
- d. It doesn’t require specifying the number of clusters

### Question 66

What is the purpose of the Silhouette Score?

Select one:

- a. To measure feature importance
- b. To evaluate the quality of clusters
- c. To select the number of neighbors
- d. To improve training accuracy

### Question 67

Which clustering technique produces a dendrogram?

Select one:

- a. K-Means
- b. DBSCAN
- c. Hierarchical Clustering
- d. Mean Shift

### Question 68

Which clustering technique uses a "core point" concept?

Select one:

- a. DBSCAN
- b. K-Means
- c. Agglomerative Clustering
- d. Factor Analysis

### Question 69

Which metric evaluates the cohesion and separation of clusters?

Select one:

- a. Davies-Bouldin Index
- b. Confusion Matrix
- c. R-Squared
- d. Mean Squared Error

### Question 70

What is a practical application of clustering?

Select one:

- a. Customer segmentation
- b. Predicting housing prices
- c. Spam detection
- d. Linear regression

---

## Correct Answers for T7

61. b
62. b
63. a
64. b
65. c
66. b
67. c
68. a
69. a
70. a

# Quiz: T8 to T10

## T8: Advanced Clustering Techniques and Soft Clustering

### Question 71

What is the primary difference between hard and soft clustering?

Select one:

- a. Hard clustering assigns data points to multiple clusters, while soft clustering assigns them to only one cluster
- b. Hard clustering assigns data points to exactly one cluster, while soft clustering assigns probabilities for belonging to multiple clusters
- c. Hard clustering uses probabilistic methods, while soft clustering uses deterministic methods
- d. Hard clustering is unsupervised, while soft clustering is supervised

### Question 72

Which clustering method assigns probabilities of cluster membership to data points?

Select one:

- a. K-Means
- b. Fuzzy C-Means
- c. DBSCAN
- d. Hierarchical Clustering

### Question 73

What is the Expectation-Maximization (EM) algorithm used for?

Select one:

- a. Calculating distances between clusters
- b. Optimizing cluster centroids in K-Means
- c. Estimating parameters for probabilistic models in clustering
- d. Determining the number of clusters in a dataset

### Question 74

Which of the following is an example of a probabilistic clustering method?

Select one:

- a. DBSCAN
- b. K-Means
- c. Gaussian Mixture Models (GMM)
- d. Agglomerative Clustering

### Question 75

What is a limitation of Fuzzy C-Means clustering?

Select one:

- a. It assumes clusters are non-spherical
- b. It requires specifying the number of clusters in advance
- c. It doesn’t handle noise or outliers well
- d. It only works with categorical data

### Question 76

Which clustering algorithm is most suitable for overlapping clusters?

Select one:

- a. K-Means
- b. Fuzzy C-Means
- c. DBSCAN
- d. Hierarchical Clustering

### Question 77

How does the EM algorithm alternate between steps?

Select one:

- a. By updating cluster centers and recalculating distances
- b. By assigning probabilities (E-step) and updating model parameters (M-step)
- c. By splitting and merging clusters iteratively
- d. By assigning data points to the nearest cluster

### Question 78

What is the main advantage of Gaussian Mixture Models (GMM) over K-Means?

Select one:

- a. GMM uses probabilistic assignments, while K-Means uses hard assignments
- b. GMM is faster to train than K-Means
- c. GMM doesn’t require specifying the number of clusters
- d. GMM is a supervised learning algorithm

### Question 79

Which clustering algorithm is based on density estimation?

Select one:

- a. Gaussian Mixture Models
- b. DBSCAN
- c. Fuzzy C-Means
- d. K-Means

### Question 80

What is the role of membership weights in Fuzzy C-Means?

Select one:

- a. To determine the number of clusters
- b. To assign data points to specific clusters
- c. To represent the degree of belonging of each data point to each cluster
- d. To measure the distance between cluster centroids

---

## Correct Answers for T8

71. b
72. b
73. c
74. c
75. b
76. b
77. b
78. a
79. b
80. c

---

## T9: Factor Analysis and Dimensionality Reduction

### Question 81

What is the primary goal of factor analysis?

Select one:

- a. To reduce dimensionality by finding independent components
- b. To identify latent variables that explain observed variables
- c. To cluster data points into distinct groups
- d. To separate noise from the signal in data

### Question 82

What is the main difference between PCA and factor analysis?

Select one:

- a. PCA focuses on covariance, while factor analysis focuses on variance
- b. PCA identifies latent variables, while factor analysis focuses on principal components
- c. PCA explains total variance, while factor analysis explains shared variance
- d. PCA is a probabilistic model, while factor analysis is deterministic

### Question 83

Which method is commonly used to compute latent variables in factor analysis?

Select one:

- a. Eigenvalue decomposition
- b. Singular Value Decomposition (SVD)
- c. Expectation-Maximization
- d. Gradient Descent

### Question 84

What does the eigenvector associated with the largest eigenvalue represent in PCA?

Select one:

- a. The direction of maximum variance in the data
- b. The center of the dataset
- c. The smallest principal component
- d. The noise component in the data

### Question 85

What is the purpose of rotation in factor analysis?

Select one:

- a. To improve interpretability of factors
- b. To reduce the number of factors
- c. To align factors with principal components
- d. To scale the data to a standard range

### Question 86

Which of the following methods is used to reduce high-dimensional data while preserving variance?

Select one:

- a. t-SNE
- b. PCA
- c. LDA
- d. Factor Analysis

### Question 87

What is a limitation of factor analysis?

Select one:

- a. It cannot handle missing data
- b. It assumes factors are uncorrelated
- c. It requires the number of factors to be pre-specified
- d. It only works for categorical data

### Question 88

Which of the following dimensionality reduction techniques is best suited for visualizing data in 2D?

Select one:

- a. PCA
- b. ICA
- c. t-SNE
- d. Factor Analysis

### Question 89

What is the difference between PCA and ICA?

Select one:

- a. PCA finds uncorrelated components, while ICA finds independent components
- b. PCA uses rotation, while ICA does not
- c. PCA focuses on shared variance, while ICA focuses on covariance
- d. PCA works on non-linear data, while ICA works only on linear data

### Question 90

What is the primary output of factor analysis?

Select one:

- a. A set of orthogonal principal components
- b. A set of latent variables
- c. A similarity matrix
- d. A clustering of data points

---

## Correct Answers for T9

81. b
82. c
83. a
84. a
85. a
86. b
87. c
88. c
89. a
90. b

---

## T10: Visualization Techniques

### Question 91

What is the primary goal of visualization in machine learning?

Select one:

- a. To improve model accuracy
- b. To interpret and analyze high-dimensional data
- c. To cluster similar data points
- d. To optimize hyperparameters

### Question 92

Which technique maps high-dimensional data into a lower-dimensional grid while preserving topology?

Select one:

- a. PCA
- b. t-SNE
- c. SOM (Self-Organizing Maps)
- d. UMAP

### Question 93

What does Multi-Dimensional Scaling (MDS) aim to preserve?

Select one:

- a. Variance in data
- b. Dissimilarity between data points
- c. Cluster centers
- d. Probabilities of cluster membership

### Question 94

Which visualization technique uses neighborhood relationships to embed data?

Select one:

- a. PCA
- b. MDS
- c. t-SNE
- d. Factor Analysis

### Question 95

What is a drawback of t-SNE compared to UMAP?

Select one:

- a. t-SNE doesn’t handle categorical data
- b. t-SNE is computationally more expensive
- c. t-SNE assumes data is linear
- d. t-SNE doesn’t work on high-dimensional data

### Question 96

Which of the following is a feature of Self-Organizing Maps (SOM)?

Select one:

- a. Supervised learning technique
- b. Preserves topological structure of data
- c. Focuses on maximizing variance
- d. Outputs probabilistic cluster memberships

### Question 97

What is a key advantage of MDS over PCA?

Select one:

- a. MDS is faster to compute
- b. MDS can work with non-linear dissimilarity measures
- c. MDS doesn’t require scaling of data
- d. MDS doesn’t reduce dimensionality

### Question 98

Which visualization technique works by projecting data onto principal components?

Select one:

- a. PCA
- b. t-SNE
- c. SOM
- d. UMAP

### Question 99

What is a limitation of MDS?

Select one:

- a. It assumes data is Gaussian
- b. It requires a dissimilarity matrix
- c. It cannot work with large datasets
- d. It doesn’t support categorical data

### Question 100

Which visualization method is best for interactive exploration of clusters in data?

Select one:

- a. PCA
- b. t-SNE
- c. UMAP
- d. All of the above

---

## Correct Answers

91. b
92. c
93. b
94. c
95. b
96. b
97. b
98. a
99. b
100. d
# Quiz: T11 to T12

## T11: Recommender Systems

### Question 101

What is the primary goal of a recommender system?

Select one:

- a. To cluster users based on their preferences
- b. To predict items a user may like based on their history
- c. To classify items into categories
- d. To analyze the content of items

### Question 102

Which of the following techniques is used in collaborative filtering?

Select one:

- a. User-user similarity
- b. Item-item similarity
- c. Both a and b
- d. None of the above

### Question 103

What is a limitation of collaborative filtering?

Select one:

- a. It cannot handle new users with no history (cold-start problem)
- b. It only works with content-based features
- c. It doesn’t require user feedback
- d. It performs poorly with small datasets

### Question 104

Which recommender system technique uses features of the items to recommend similar items?

Select one:

- a. Collaborative filtering
- b. Content-based filtering
- c. Hybrid filtering
- d. Group recommender systems

### Question 105

What is the primary advantage of hybrid recommender systems?

Select one:

- a. They eliminate the need for user data
- b. They combine strengths of multiple techniques to improve accuracy
- c. They require less computational power
- d. They only use collaborative filtering

### Question 106

Which of the following is an evaluation metric for recommender systems?

Select one:

- a. Precision
- b. Recall
- c. F1 Score
- d. All of the above

### Question 107

What is a conversational recommender system?

Select one:

- a. A system that uses clustering to group items
- b. A system that engages in dialogue with users to refine recommendations
- c. A system that only works with explicit feedback
- d. A system that assigns probabilities to user preferences

### Question 108

Which of the following describes the "cold-start problem"?

Select one:

- a. A system’s inability to recommend unpopular items
- b. Difficulty in recommending items to new users with no history
- c. The system’s inability to handle sparse data
- d. Difficulty in scaling to large datasets

### Question 109

What is the primary challenge in group recommender systems?

Select one:

- a. Combining preferences of multiple users
- b. Handling large datasets
- c. Evaluating precision and recall
- d. Scaling recommendations in real time

### Question 110

What is a key characteristic of content-based recommender systems?

Select one:

- a. They use user-user similarities
- b. They analyze item features to recommend similar items
- c. They require data from other users
- d. They cluster users based on their ratings

---

## Correct Answers for T11

101. b
102. c
103. a
104. b
105. b
106. d
107. b
108. b
109. a
110. b

---

## T12: Bias and Fairness in Machine Learning

### Question 111

What is "bias" in the context of machine learning?

Select one:

- a. The tendency of a model to favor certain outcomes unfairly
- b. An error due to high model complexity
- c. The process of selecting important features
- d. The variance in model predictions across datasets

### Question 112

What is "fairness" in machine learning?

Select one:

- a. Ensuring that models are unbiased
- b. Providing equal predictive performance across groups
- c. Achieving the highest accuracy possible
- d. Avoiding overfitting in the training phase

### Question 113

Which of the following is an example of a biased dataset?

Select one:

- a. A dataset where all features are normalized
- b. A dataset with overrepresentation of one demographic group
- c. A dataset with missing values
- d. A dataset with more categorical features than numerical ones

### Question 114

What is the "cold-start" problem’s relationship to fairness?

Select one:

- a. It affects how recommender systems treat new users
- b. It ensures that underrepresented groups are equally represented
- c. It measures fairness through cross-validation
- d. It prevents data leakage in model training

### Question 115

Which of the following is a method to mitigate bias in data?

Select one:

- a. Regularization
- b. Data augmentation
- c. Sampling strategies to balance class representation
- d. Hyperparameter tuning

### Question 116

What does "anti-classification" in fairness refer to?

Select one:

- a. Ensuring protected attributes are not used in predictions
- b. Maximizing model accuracy
- c. Regularizing the model to reduce overfitting
- d. Allowing predictions based on sensitive attributes

### Question 117

What is "calibration" in the context of fairness?

Select one:

- a. Ensuring equal predictive probabilities across groups
- b. Adjusting feature scaling to improve accuracy
- c. Regularizing model coefficients
- d. Balancing the dataset to avoid bias

### Question 118

What is a real-world example of bias in machine learning?

Select one:

- a. Gender bias in hiring algorithms
- b. Overfitting in training models
- c. Poor performance on validation datasets
- d. Low precision in spam detection systems

### Question 119

What is "representation bias"?

Select one:

- a. Errors caused by poor model regularization
- b. When certain groups are overrepresented or underrepresented in training data
- c. The inability of a model to generalize across datasets
- d. Variance in predictions due to noise in data

### Question 120

Which fairness approach requires equal performance metrics across groups?

Select one:

- a. Calibration
- b. Classification parity
- c. Anti-classification
- d. Data normalization

---

## Correct Answers for T12

111. a
112. b
113. b
114. a
115. c
116. a
117. a
118. a
119. b
120. b