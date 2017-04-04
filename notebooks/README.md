# Notebook topics

## 01 Introduction to Machine Learning.ipynb
General machine learning review, no code.

## 02 Scientific Computing Tools in Python.ipynb
Intro to Jupyter Notebooks, NumPy, SciPy sparse matrices, Matplotlib

## 03 Data Representation for Machine Learning.ipynb
Intro to loading different datasets
  - iris
  - NIST Digits
  - S-Curve
  - Olivette faces

## 04 Training and Testing Data.ipynb
Discusion on separating training data into a training and test set.  Some important steps
  - `sklearn.cross_validation.train_test_split`: randomize the data before splitting
  - pass `stratify=y`: keep an equal distribution of different classes

## 05 Supervised Learning - Classification.ipynb
Demonstrate machine learning with 2D data (`sklearn.datasets.make_blobs`), to simplify visualization.  Use the following classifiers:
  1 [`LogisticRegression`](http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
  2 [`KNeighborsClassifier`](http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html) (note too few neighbors leads to overfitting)

## 06 Supervised Learning - Regression.ipynb
Present regression, predicting a continuous output variable.
  1. LinearRegression
  2. KNeighborsRegression

## 07 Unsupervised Learning - Transformations and Dimensionality Reduction.ipynb
Key points about unlabeled data:
  1. Standardization: $\x_\text{standardized} = \frac{x - \mu_x}{\sigma_x}$ where $\mu_x$ and $\sigma_x$ are the mean and standard deviation respectively
    a. Standard Scaler: depends heavily on the mean which can be skewed by outliers
    b. Robust Scaler: uses the quartiles/median to avoid the effects of outliers
  2. Data Reduction using Principal Component Analysis (PCA): identify the eigenvectors (new features) of the data, then identify which can described the majority of the variation within the data. Demonstration using MNIST dataset.

## 08 Unsupervised Learning - Clustering.ipynb
Demonstrate different clustering algorithms using the `make_blobs` dataset.  Code uses `KMeans`: must specify number of clusters, assumes the clusters have equal spherical variance (demonstrates the elbow method for selecting the number of clusters to use).  There are illustrations of several others, each with their own assumptions.

## 09 Review of Scikit-learn API.ipynb
Recap (text summary) of the API.  Interesting items:
  - `model.predict_proba()` : For supervised classification problems, some estimators also provide this method, which returns the probability that a new observation has each categorical label. In this case, the label with the highest probability is returned by `model.predict()`.

## 10 Case Study - Titanic Survival.ipynb
Feature extraction.
  - numerical
  - categorical

## 11 Text Feature Extraction.ipynb

## 12 Case Study - SMS Spam Detection.ipynb

## 13 Cross Validation.ipynb

## 14 Model Complexity and GridSearchCV.ipynb

## 15 Pipelining Estimators.ipynb

## 16 Performance metrics and Model Evaluation.ipynb

## 17 In Depth - Linear Models.ipynb

## 18 In Depth - Support Vector Machines.ipynb

## 19 In Depth - Trees and Forests.ipynb

## 20 Feature Selection.ipynb

## 21 Unsupervised learning - Hierarchical and density-based clustering algorithms.ipynb

## 22 Unsupervised learning - Non-linear dimensionality reduction.ipynb

## 23 Out-of-core Learning Large Scale Text Classification.ipynb
