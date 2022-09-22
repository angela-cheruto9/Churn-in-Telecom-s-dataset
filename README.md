# Churn-in-Telecom-s-dataset

<img src="https://d1.awsstatic.com/customer-references-customer-content/AdobeStock_434719793_BAE.687d9b655f7eb111db46a4401271b519be59bef0.jpeg" width="1000" height="600">

# Project aim
The aim of this project is to retain more customers by looking into what factors are mostly attached to customers unsubscribing to a service in this case it is telecommunication service.

# Libraries
Python

Pandas

Matplotlib

Seaborn

Scikit-Learn

# Selection of data

### loading data
df = pd.read_csv('churn.csv')
* Importing necessary libraries
* Loading dataset

### Exploratory data analysis
* Checking the shape of the dataset
* Checking the datatypes in the dataset
* Descriptive statistics of the dataset

### Datatype conversion
* Detecting and dealing with missing values


# Data preprocessing
* Checking for and removing multicollinearity (correlated predictors)
* Identifying numerical features
* Identifying categorical features
* Visualization of the numerical features
* Analysis of numerical variables against churn
* Visualization of the categorical features
* Univariate analysis
* Distribution of churn(target variable) in the dataset
* Bivariate analysis of categorical variables against churn

# Data transformation
* Joining categorical and numerical data
* Joining categorical data to numerical data
* Converting categorical data into numerical
* Converting the categorical variables into numerical and avoiding dummy varibale trap

### Summary of visualizations

After visualizing the numerical and categorical features in our dataset, some few steps can then be taken:

1.Standardizing numerical data

2.Converting categorical data into numerical data in order to enable building of a machine learning model

3.Dealing with some of the features that are skewed or have imbalanced data 

### Perfoming train_test_split
### Handling imbalanced data

### Standardizing continuous variables
* Standardizing the Dataset


# Building a model
* Fitting train data
* Predictions
* Getting the precision,accuracy,recall and f1_score

* Random forest classifier
* SVM
* KNeighbors Classifier
* Decision Tree Classifier
* Gradient Boosting Classifier
* Logistic regression
* Performance of model

### Pickling the model

* Assigning a name to the model
* Dumping the created model for later use
* Calling the file 
* Checking score of loaded model

### Intepretation and evaluation










