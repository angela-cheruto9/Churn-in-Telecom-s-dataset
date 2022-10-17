# Churn-in-Telecom-s-dataset

<img src="https://d1.awsstatic.com/customer-references-customer-content/AdobeStock_434719793_BAE.687d9b655f7eb111db46a4401271b519be59bef0.jpeg" width="1000" height="600">

# Project aim
The aim of this project is to retain more customers by looking into what factors are mostly attached to customers unsubscribing to a service in this case it is telecommunication service.

The data used is obtained from [Kaggle](https://www.kaggle.com/datasets/becksddf/churn-in-telecoms-dataset)

## Project aim

Build a classification model that predict whether a customer will pull out of the telecommunication service or not based on several factors lke state,area code,voice mail plan,charges,calls, minutes, international plan and churn.

The target variable(churn) will be used to create a model that can determine whether a customer will churn or not by using binary 1 and 0 (1 means the loss of client, i.e. churn, 0 means no loss of client).

## Data

The data used is obtained from [Kaggle](https://www.kaggle.com/datasets/becksddf/churn-in-telecoms-dataset),which can also be found in the data folder in this project's GitHub repository.
The dataset contains 3333 rows and 21 columns.

## Defining Experimental design

* Importing the relevant libraries used in the analysis.

* Loading data

* Exploratory Data Analysis (EDA)

* Data Pre-processing

* Modelling and Evaluation

* Challenging the model

* Conclusion

* Recommendations


## Libraries
Python

Pandas

Matplotlib

Seaborn

Scikit-Learn

## Data exploration

![Screenshot (387)](https://user-images.githubusercontent.com/104419035/196142058-187c8132-cab0-426d-9039-94db1a572508.png)

The False count added up to 2850 and True count was 483. False meaning the client did not churn and True meaning the client churned.

Data is highly imbalanced, as the ratio of the customers who churn and those who don't churn is 86:14 . I then analysed the data with other features while taking the target value into consideration.

## Data preprocessing

The datasets contains customers who churned and those who did not.
We started by detecting and dealing with missing values, datatype conversion, checking and removing multicollinearity,analysis of variables against the target variable(churn) and visualization of both numerical and categorical variables.

### Summary of visualizations

After visualizing the numerical and categorical features in our dataset, some few steps were taken:

1.Standardizing numerical data.

2.Converting categorical data into numerical data in order to enable building of a machine learning model.

3.Dealing with some of the features that are skewed or have imbalanced data.


![Screenshot (388)](https://user-images.githubusercontent.com/104419035/196146106-8459746f-2591-4048-ad98-1e6874513ac5.png)

Churn is high for those with international plans, those who make several customer service calls especially more than 4 the churn rate seems to be pretty high.

Clients who make day calls also have a high churn rate than clients who make night calls this could probably be the charge rate for day calls being higher.

Churn is low for clients with a voice mail plan,with more number of voice mail messages, and those clients with a high number of international calls.

## Modelling and evaluation

I performed a train-test-split and asssigned y to our target variable.

After building several classification models, Support Vector Machine classification model had the highest precision.

![Screenshot (389)](https://user-images.githubusercontent.com/104419035/196152248-0b5b4293-795b-4c2d-a36a-f4408380bad0.png)

The train data had an accuracy score of 90% and a precision score of 97%

SVM is the preffered model because the project aims to find out if there are any predictable patterns of the customers unsubcribing or leaving the company service, therefore, we maximize our true positives.

Precision was used as our error metric for our algorithm, which is true positives / (false positives + true positives),which ensured that we minimize how much money we lose with false positives.We'd rather minimize our potential losses than maximize our potential gains. We are not overfitting the model because the accuracy score of the training data isn't far from the accuracy score of the test data.

### Pickling the model

Assigned the model a name and dumped it for later use.Checked the score of loaded model.

### Intepretation and evaluation










