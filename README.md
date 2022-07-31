# Churn-in-Telecom-s-dataset

from IPython.display import Image
Image(url='https://d1.awsstatic.com/customer-references-customer-content/AdobeStock_434719793_BAE.687d9b655f7eb111db46a4401271b519be59bef0.jpeg')

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
*Importing necessary libraries*
*loading dataset*

### Exploratory data analysis
*check the shape in the dataset*
* check the datatypes in the dataset*
*descriptive statistics of the dataset*

### Datatype conversion
*Detecting and dealing with missing values*


# Data preprocessing
*Checking for and removing multicollinearity (correlated predictors)*
*Identifying numerical features*
*Identifying categorical features*
*Visualization of the numerical features*
*Analysis of numerical variables against churn*
*countplot to show the distribution of customer service calls against churn in the dataset*
*countplot to show the distribution of total intl calls against churn in the dataset*
*Visualization of the categorical features*
* Univariate analysis*
*countplot to show the distribution of churn(target variable) in the dataset*
*Bivariate analysis of categorical variables against churn*

# Data transformation

*Joining categorical and numerical data*
*joining categorical data to numerical data*

*Converting categorical data into numerical*
*Converting the categorical variables into numerical and avoiding dummy varibale trap*
*KDE plots to represent features in the dataset*
 
*checking the list of columns using for loop*

### Summary of visualizations

After visualizing the numerical and categorical features in our dataset, some few steps can then be taken:

1.Normalize numerical data

2.Convert categorical data into numerical data in order to enable building of a machine learning model

3.Deal with some of the features that are skewed or have imbalanced data 

### Perfoming train_test_split
### Handling imbalanced data

### Standardizing continuous variables
*Standardizing the Dataset*


# Building a model
fitting train data
predicting 
getting the precision,accuracy,recall and f1_score

### Random forest classifier

### SVM
### KNeighbors Classifier

### Decision Tree Classifier

### Gradient Boosting Classifier

### Logistic regression

### Performance on Test Data

### Pickling the model

*assign a name to the model*
*dump the created model for later use*
*call the file to check if it was successfully loaded*
*check score of loaded model*

### Intepretation and evaluation










