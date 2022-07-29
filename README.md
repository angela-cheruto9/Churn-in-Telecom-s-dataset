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
df.shape

* check the datatypes in the dataset*
df.info()

*descriptive statistics of the dataset*
df.describe()

### Datatype conversion
*Detecting and dealing with missing values*
df['international plan'] = df['international plan'].map({'no': 0, 'yes': 1})
df['voice mail plan'] = df['voice mail plan'].map({'no': 0, 'yes': 1})
df['churn'] = df['churn'].astype(int)


# Data preprocessing
*Checking for and removing multicollinearity (correlated predictors)*
abs(df.corr()) > 0.75

*Identifying numerical features*
num = df.select_dtypes(exclude=['object', 'bool'])
num.head()

*Identifying categorical features*
cat = df.select_dtypes(exclude=['number', 'float'])
cat

*Visualization of the numerical features*
num.hist(figsize=(15,15), bins='auto');

*Analysis of numerical variables against churn*
*countplot to show the distribution of customer service calls against churn in the dataset*
sns.set_theme(style="darkgrid")
sns.countplot(x='customer service calls', hue='churn', data= df)

*countplot to show the distribution of total intl calls against churn in the dataset*
sns.set_theme(style="darkgrid")
sns.countplot(x='total intl calls', hue='churn', data= df)

*Visualization of the categorical features*
* Univariate analysis*
*countplot to show the distribution of churn(target variable) in the dataset*
sns.set_theme(style="darkgrid")
ax = sns.countplot(x="churn", data= cat)

*Bivariate analysis of categorical variables against churn*
for i, predictor in enumerate(cat.drop(columns=['churn'])):
    plt.figure(i)
    sns.countplot(data= cat, x=predictor, hue='churn')

# Data transformation

*Joining categorical and numerical data*
*joining categorical data to numerical data*
df_num_cat = pd.concat([num, cat], axis=1) 
df_num_cat.head()

*Converting categorical data into numerical*
*Converting the categorical variables into numerical and avoiding dummy varibale trap*
df_num_cat  = pd.get_dummies(df_num_cat ,drop_first=True)
df_num_cat.head()

*KDE plots to represent features in the dataset*
*# from our normalized data create a list of columns*
data = list(df_num_cat) 
*checking the list of columns using for loop*
for column in data:
    # create a histogram
    df_num_cat[column].plot.hist(density=True, label = column+' histogram' ) 
    # create a KDE plot
    df_num_cat[column].plot.kde(label =column+' kde') 
    plt.legend()
    plt.show() 

### Summary of visualizations

After visualizing the numerical and categorical features in our dataset, some few steps can then be taken:

1.Normalize numerical data

2.Convert categorical data into numerical data in order to enable building of a machine learning model

3.Deal with some of the features that are skewed or have imbalanced data 

# Building a model

### Perfoming train_test_split
### Handling imbalanced data
*importing necessary library*
from imblearn.over_sampling import SMOTE
*fit X and y to SMOTE*
X,y= SMOTE().fit_resample(X,y)
*get the value_counts of y*
y.value_counts()


### Standardizing continuous variables
*Standardizing the Dataset*
*import necessary library*
from sklearn.preprocessing import StandardScaler
*instatiate scaler*
scaler = StandardScaler()
*fit_transform X_train and X_test onto the scaler*
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


### Random forest classifier
*import necessary library*
from sklearn.ensemble import RandomForestClassifier
*instatiate the RandomForestClassifier*
rf = RandomForestClassifier()
*fit the RandomForestClassifier*
rf.fit(X_train_scaled,y_train)

print("Accuracy:", accuracy_score(y_test,y_pred))
print("Precision:", precision_score(y_test,y_pred))
print("Recall:  ", recall_score(y_test,y_pred))
print("F1_score:  ", f1_score(y_test,y_pred))
                     

### SVM
*import necessary library*
from sklearn import svm
*instatiate classifier*
svm = svm.SVC(probability=True)
*fit the model on train data*
svm.fit(X_train_scaled, y_train)

print("Accuracy:", accuracy_score(y_test,y_pred1))
print("Precision:", precision_score(y_test,y_pred1))
print("Recall:  ", recall_score(y_test,y_pred1))
print("F1_score:  ", f1_score(y_test,y_pred1))            

### KNeighbors Classifier
*import relevant library*
from sklearn.neighbors import KNeighborsClassifier
*instatiate classifier*
knn =  KNeighborsClassifier()
*fit train data to the classifier*
knn.fit(X_train_scaled, y_train)

print("Accuracy:", accuracy_score(y_test,y_pred2))
print("Precision:", precision_score(y_test,y_pred2))
print("Recall:  ", recall_score(y_test,y_pred2))
print("F1_score:  ", f1_score(y_test,y_pred2))
                     
### Decision Tree Classifier
*import necessary library*
from sklearn.tree import DecisionTreeClassifier
*instatiate classifier*
dt = DecisionTreeClassifier(criterion = "gini",random_state = 100,max_depth=5, min_samples_leaf=8)
*fit the train data to dt*
dt.fit(X_train_scaled,y_train)


print("Accuracy:", accuracy_score(y_test,y_pred3))
print("Precision:", precision_score(y_test,y_pred3))
print("Recall:  ", recall_score(y_test,y_pred3))
print("F1_score:  ", f1_score(y_test,y_pred3))


### Gradient Boosting Classifier
*import relevant library*
from sklearn.ensemble import GradientBoostingClassifier
*instatiate classifier*
gbc= GradientBoostingClassifier()
*fit X_train to the model*
gbc.fit(X_train_scaled,y_train)

print("Accuracy:", accuracy_score(y_test,y_pred4))
print("Precision:", precision_score(y_test,y_pred4))
print("Recall:  ", recall_score(y_test,y_pred4))
print("F1_score:  ", f1_score(y_test,y_pred4))

### Logistic regression
*import library*
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
*Fit the logistic Regression Model*
logmodel = LogisticRegression(random_state=50)
logmodel.fit(X_train_scaled,y_train)
*Predict the value for new, unseen data*
y_pred5 = logmodel.predict(X_test_scaled)
*Find Accuracy using accuracy_score method*
logmodel_accuracy = metrics.accuracy_score(y_test, y_pred5) 
logmodel_accuracy


print("Accuracy:", accuracy_score(y_test,y_pred5))
print("Precision:", precision_score(y_test,y_pred5))
print("Recall:  ", recall_score(y_test,y_pred5))
print("F1_score:  ", f1_score(y_test,y_pred5))



### Performance on Test Data
*import necessary library*
from sklearn import svm
*instatiate classifier*
svm = svm.SVC(probability=True)
*fit the model on test data*
svm.fit(X_test_scaled, y_test)

print("Accuracy:", accuracy_score(y_test,y_hat_test ))
print("Precision:", precision_score(y_test,y_hat_test ))
print("Recall:  ", recall_score(y_test,y_hat_test))
print("F1_score:  ", f1_score(y_test,y_hat_test))
                     
### Pickling the model
 import library
import pickle
*assign a name to the model*
filename = 'churn_model'
*dump the created model for later use*
pickle.dump(svm, open(filename, 'wb'))

*call the file to check if it was successfully loaded*
load_model = pickle.load(open(filename, 'rb'))

*check score of loaded model*
model_score_r1 = load_model.score(X_test_scaled, y_test)
model_score_r1









