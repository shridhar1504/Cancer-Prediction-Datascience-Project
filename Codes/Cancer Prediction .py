#!/usr/bin/env python
# coding: utf-8

#    # Cancer Prediction

# "Importing the required libraries & packages"

# In[1]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split,cross_val_score,KFold,GridSearchCV
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
import pickle
import warnings
warnings.filterwarnings('ignore')


# "Changing The Default Working Directory Path 
# &
# Reading the Dataset using Pandas Command"

# In[2]:


os.chdir('C:\\Users\\Shridhar\\OneDrive\\Desktop\\Top Mentor\\Batch 74 Day 31\\Project to Explain Classifciation')
df=pd.read_csv('data.csv')


# "Checking the Null values of all the columns in the dataset."

# In[3]:


df.isna().sum()


# "Getting the unique value and its counts in the Diagnosis Column from the dataset"

# In[4]:


df.diagnosis.value_counts()


# "Label Encoding the Diagnosis Column from the dataset."

# In[5]:


df['diagnosis']=df['diagnosis'].astype('category')
df['diagnosis']=df['diagnosis'].cat.codes


# "Assigning the dependent and independent variable."

# In[6]:


x=df.drop(['id','diagnosis'],axis=1)
y=df['diagnosis']


# "Getting the Correlation Values from all the numeric columns from the independent variable using Seaborn Heatmap & saving the PNG File"

# In[7]:


plt.rcParams['figure.figsize']=20,12
sns.heatmap(x.corr(),annot=True)
plt.title('Correlation Heat Map')
plt.savefig('Correlation Heat Map.png')
plt.show()


# "Checking the Outliers of all the columns of the independent variable using Seaborn Box Plot in the following 6 Cells"

# In[8]:


f, (ax1,ax2,ax3,ax4,ax5) = plt.subplots (1,5) 
sns.boxplot ( x= df['diagnosis'], y = df['radius_mean'], ax = ax1)
sns.boxplot (x= df['diagnosis'], y = df['texture_mean'], ax = ax2)
sns.boxplot (x= df['diagnosis'], y = df['perimeter_mean'], ax = ax3)
sns.boxplot (x= df['diagnosis'], y = df['area_mean'] , ax = ax4)
sns.boxplot (x= df['diagnosis'], y = df['smoothness_mean']  , ax = ax5)
f .tight_layout()


# In[9]:


f, (ax1,ax2,ax3,ax4,ax5) = plt.subplots (1,5)
sns.boxplot (x= df['diagnosis'], y = df['compactness_mean'], ax = ax1)
sns.boxplot (x= df['diagnosis'], y = df['concavity_mean'] , ax = ax2)
sns.boxplot (x= df['diagnosis'], y = df['concave points_mean'] , ax = ax3)
sns.boxplot (x= df['diagnosis'], y = df['symmetry_mean'], ax = ax4)
sns.boxplot (x= df['diagnosis'], y = df['fractal_dimension_mean'] , ax = ax5)
f .tight_layout()


# In[10]:


f, (ax1,ax2,ax3,ax4,ax5) = plt.subplots (1,5)
sns.boxplot (x= df['diagnosis'], y = df['radius_se'], ax = ax1)
sns.boxplot (x= df['diagnosis'], y = df['texture_se'] , ax = ax2)
sns.boxplot (x= df['diagnosis'], y = df['perimeter_se'] , ax = ax3)
sns.boxplot (x= df['diagnosis'], y = df['area_se'], ax = ax4)
sns.boxplot (x= df['diagnosis'], y = df['smoothness_se'] , ax = ax5)
f .tight_layout()


# In[11]:


f, (ax1,ax2,ax3,ax4,ax5) = plt.subplots (1,5)
sns.boxplot (x= df['diagnosis'], y = df['compactness_se'], ax = ax1)
sns.boxplot (x= df['diagnosis'], y = df['concavity_se'] , ax = ax2)
sns.boxplot (x= df['diagnosis'], y = df['concave points_se'] , ax = ax3)
sns.boxplot (x= df['diagnosis'], y = df[ 'symmetry_se'], ax = ax4)
sns.boxplot (x= df['diagnosis'], y = df['fractal_dimension_se'] , ax = ax5)
f .tight_layout()


# In[12]:


f, (ax1,ax2,ax3,ax4,ax5) = plt.subplots (1,5)
sns.boxplot (x= df['diagnosis'], y = df['radius_worst'], ax = ax1)
sns.boxplot (x= df['diagnosis'], y = df['texture_worst'] , ax = ax2)
sns.boxplot (x= df['diagnosis'], y = df['perimeter_worst'] , ax = ax3)
sns.boxplot (x= df['diagnosis'], y = df['area_worst'], ax = ax4)
sns.boxplot (x= df['diagnosis'], y = df['smoothness_worst'] , ax = ax5)
f .tight_layout()


# In[13]:


f, (ax1,ax2,ax3,ax4,ax5) = plt.subplots (1,5)
sns.boxplot (x= df['diagnosis'], y = df['compactness_worst'], ax = ax1)
sns.boxplot (x= df['diagnosis'], y = df['concavity_worst'] , ax = ax2)
sns.boxplot (x= df['diagnosis'], y = df['concave points_worst'] , ax = ax3)
sns.boxplot (x= df['diagnosis'], y = df[ 'symmetry_worst'], ax = ax4)
sns.boxplot (x= df['diagnosis'], y = df['fractal_dimension_worst'] , ax = ax5)
f .tight_layout()


# "Visualizing the data distribution of all the columns in independent variable against the density distribution using Seaborn Distplot in the following 6 Cells.

# In[14]:


g = sns.FacetGrid (df,col = 'diagnosis', hue = 'diagnosis')
g.map (sns.distplot, "radius_mean", hist = False, rug = True)

g = sns.FacetGrid (df,col = 'diagnosis', hue = 'diagnosis')
g.map (sns.distplot, 'texture_mean', hist = False, rug = True)

g = sns.FacetGrid (df,col = 'diagnosis', hue = 'diagnosis')
g.map (sns.distplot, 'perimeter_mean', hist = False, rug = True)

g = sns.FacetGrid (df,col = 'diagnosis', hue = 'diagnosis')
g.map (sns.distplot, "area_mean", hist = False, rug = True)

g = sns.FacetGrid (df,col = 'diagnosis', hue = 'diagnosis')
g.map (sns.distplot, "smoothness_mean", hist = False, rug = True)
plt.show()


# In[15]:


g = sns.FacetGrid (df,col = 'diagnosis', hue = 'diagnosis')
g.map (sns.distplot, "compactness_mean", hist = False, rug = True)

g = sns.FacetGrid (df,col = 'diagnosis', hue = 'diagnosis')
g.map (sns.distplot, "concavity_mean", hist = False, rug = True)

g = sns.FacetGrid (df,col = 'diagnosis', hue = 'diagnosis')
g.map (sns.distplot, "concave points_mean", hist = False, rug = True)

g = sns.FacetGrid (df,col = 'diagnosis', hue = 'diagnosis')
g.map (sns.distplot, "symmetry_mean", hist = False, rug = True)

g = sns.FacetGrid (df,col = 'diagnosis', hue = 'diagnosis')
g.map (sns.distplot, "fractal_dimension_mean", hist = False, rug = True)
plt.show()


# In[16]:


g = sns.FacetGrid (df,col = 'diagnosis', hue = 'diagnosis')
g.map (sns.distplot, "radius_se", hist = False, rug = True)

g = sns.FacetGrid (df,col = 'diagnosis', hue = 'diagnosis')
g.map (sns.distplot, 'texture_se', hist = False, rug = True)

g = sns.FacetGrid (df,col = 'diagnosis', hue = 'diagnosis')
g.map (sns.distplot, 'perimeter_se', hist = False, rug = True)

g = sns.FacetGrid (df,col = 'diagnosis', hue = 'diagnosis')
g.map (sns.distplot, "area_se", hist = False, rug = True)

g = sns.FacetGrid (df,col = 'diagnosis', hue = 'diagnosis')
g.map (sns.distplot, "smoothness_se", hist = False, rug = True)
plt.show()


# In[17]:


g = sns.FacetGrid (df,col = 'diagnosis', hue = 'diagnosis')
g.map (sns.distplot, "compactness_se", hist = False, rug = True)

g = sns.FacetGrid (df,col = 'diagnosis', hue = 'diagnosis')
g.map (sns.distplot, "concavity_se", hist = False, rug = True)

g = sns.FacetGrid (df,col = 'diagnosis', hue = 'diagnosis')
g.map (sns.distplot, "concave points_se", hist = False, rug = True)

g = sns.FacetGrid (df,col = 'diagnosis', hue = 'diagnosis')
g.map (sns.distplot, "symmetry_se", hist = False, rug = True)

g = sns.FacetGrid (df,col = 'diagnosis', hue = 'diagnosis')
g.map (sns.distplot, "fractal_dimension_se", hist = False, rug = True)
plt.show()


# In[18]:


g = sns.FacetGrid (df,col = 'diagnosis', hue = 'diagnosis')
g.map (sns.distplot, "radius_worst", hist = False, rug = True)

g = sns.FacetGrid (df,col = 'diagnosis', hue = 'diagnosis')
g.map (sns.distplot, 'texture_worst', hist = False, rug = True)

g = sns.FacetGrid (df,col = 'diagnosis', hue = 'diagnosis')
g.map (sns.distplot, 'perimeter_worst', hist = False, rug = True)

g = sns.FacetGrid (df,col = 'diagnosis', hue = 'diagnosis')
g.map (sns.distplot, "area_worst", hist = False, rug = True)

g = sns.FacetGrid (df,col = 'diagnosis', hue = 'diagnosis')
g.map (sns.distplot, "smoothness_worst", hist = False, rug = True)
plt.show()


# In[19]:


g = sns.FacetGrid (df,col = 'diagnosis', hue = 'diagnosis')
g.map (sns.distplot, "compactness_worst", hist = False, rug = True)

g = sns.FacetGrid (df,col = 'diagnosis', hue = 'diagnosis')
g.map (sns.distplot, "concavity_worst", hist = False, rug = True)

g = sns.FacetGrid (df,col = 'diagnosis', hue = 'diagnosis')
g.map (sns.distplot, "concave points_worst", hist = False, rug = True)

g = sns.FacetGrid (df,col = 'diagnosis', hue = 'diagnosis')
g.map (sns.distplot, "symmetry_worst", hist = False, rug = True)

g = sns.FacetGrid (df,col = 'diagnosis', hue = 'diagnosis')
g.map (sns.distplot, "fractal_dimension_worst", hist = False, rug = True)
plt.show()


# "Defining the Function for the ML algorithms using GridSearchCV Algorithm and splitting the dependent variable & independent variable into training and test dataset and Predicting the Dependent Variable by fitting the given model and create the pickle file of the model with the given Algo_name. Further getting the Best Parameters of the algorithm, Accuracy Score, Classification Report and Confusion Matrix between the predicted values and dependent test dataset." 

# In[20]:


def FitModel (x,y, algo_name , algorithm, gridSearchParams, cv):
    np.random.seed(10)
    x_train, x_test, y_train, y_test = train_test_split (x,y,test_size = 0.2)
    grid = GridSearchCV(estimator = algorithm, param_grid = gridSearchParams,
                        cv = cv, scoring = 'accuracy', verbose = 1 , n_jobs = -1 )
    grid_result = grid.fit(x_train, y_train)
    best_params = grid_result.best_params_
    pred = grid_result.predict (x_test)
    cm = confusion_matrix (y_test,pred) 
    pickle.dump(grid_result,open(algo_name,'wb'))
    print ('Best Params :', best_params)
    print ('\n Classification Report:',classification_report(y_test,pred))
    print ('\n Accuracy Score', (accuracy_score(y_test,pred)))
    print ('\n Confusion Matrix :\n',cm)


# "Running the function with some appropriate parameters and fitting the Support Vector Machine Classifiers Algorithm and getting the Best Parameters of the algorithm, Accuracy Score, Classification Report and Confusion Matrix between the predicted values and dependent test dataset and also the pickle file with the name SVC." 

# In[21]:


param = {'C': [0.1,1,100,1000],
        'gamma':[0.0001,0.001, 0.005, 0.1,1, 3,5,10, 100]
         }
FitModel (x,y,'SVC',SVC(), param, cv =10)


# "Running the function with some appropriate parameters and fitting the Random Forest Classifiers Algorithm and getting the Best Parameters of the algorithm, Accuracy Score, Classification Report and Confusion Matrix between the predicted values and dependent test dataset and also the pickle file with the name Random Forest."

# In[22]:


param={'n_estimators':[100,200,300,500,1000,2000],
      'criterion':['entropy','gini']}
FitModel(x,y,'Random Forest',RandomForestClassifier(),param,cv=10)


# "Running the function with some appropriate parameters and fitting the XGBoost Classifiers Algorithm and getting the Best Parameters of the algorithm, Accuracy Score, Classification Report and Confusion Matrix between the predicted values and dependent test dataset and also the pickle file with the name XGBoost."

# In[23]:


param = { 'n_estimators': [100,111,222,333,444,500,1000,2000]  }
FitModel (x,y,'XGBoost', XGBClassifier(),param, cv = 10)


# "Resampling the dependent variable so that the dependent variable values get balanced and assigning the new name for resampled variable"

# In[24]:


from imblearn.over_sampling import SMOTE
display (df['diagnosis'].value_counts())
sm = SMOTE(random_state =42)
x_res, y_res = sm.fit_resample (x, y)
display (y_res.value_counts())


# "Running the function with some appropriate parameters with the resampled variables and fitting the Random Forest Classifiers Algorithm and getting the Best Parameters of the algorithm, Accuracy Score, Classification Report and Confusion Matrix between the predicted values and dependent test dataset and also the pickle file with the name Random Forest get updated."

# In[25]:


param={'n_estimators':[100,200,300,500,1000,2000],
      'criterion':['entropy','gini']}
FitModel(x_res,y_res,'Random Forest',RandomForestClassifier(),param,cv=10)


# "Running the function with some appropriate parameters with the resampled variables and fitting the Support Vector Machine Classifiers Algorithm and getting the Best Parameters of the algorithm, Accuracy Score, Classification Report and Confusion Matrix between the predicted values and dependent test dataset and also the pickle file with the name SVC get updated."

# In[26]:


param = {'C': [0.1,1,100,1000],
        'gamma':[0.0001,0.001, 0.005, 0.1,1, 3,5,10, 100]
         }
FitModel (x_res,y_res,'SVC',SVC(), param, cv =10)


# "Running the function with some appropriate parameters with the resampled variables and fitting the XGBoost Classifiers Algorithm and getting the Best Parameters of the algorithm, Accuracy Score, Classification Report and Confusion Matrix between the predicted values and dependent test dataset and also the pickle file with the name XGBoost get updated."

# In[27]:


param = { 'n_estimators': [100,111,222,333,444,500,1000,2000]  }
FitModel (x_res,y_res,'XGBoost', XGBClassifier(),param, cv = 10)


# "Fitting the Random Forest Classifiers model with the original dependent and independent variable and getting the Accuracy Score, Classification Report and Confusion Matrix between the predicted value and dependent test dataset."

# In[28]:


np.random.seed(10)
x_train,x_test, y_train,y_test = train_test_split (x,y,test_size = 0.2)
forest = RandomForestClassifier (n_estimators = 500)
fit = forest.fit (x_train, y_train)
accuracy = fit.score(x_test,y_test)
predict = fit.predict(x_test)
cmatrix = confusion_matrix (y_test, predict)
print ('Classification Report:',classification_report(y_test,predict))
print ('Accuracy Score', (accuracy_score(y_test,predict)))
print ('Accuracy of Random Forest ', (accuracy))
print ('Confusion Matrix :\n',cmatrix)


# "Finding the feature importances of all the columns in the independent variable with respect to Random Forest Classification Model above predicted for the dimensional reduction process"

# In[29]:


importances = forest.feature_importances_
indices = np.argsort(importances)[::-1]
print ("Feature Ranking:")
for f in range (x.shape[1]):
    print ("Feature %s (%f)"  %(list (x)[f],importances[indices[f]]))


# "Fitting the Random Forest Classifiers model with the resampled dependent and independent variable and getting the Accuracy Score, Classification Report and Confusion Matrix between the predicted value and dependent test dataset."

# In[30]:


np.random.seed(10)
x_train,x_test, y_train,y_test = train_test_split (x_res,y_res,test_size = 0.2)
forest1 = RandomForestClassifier (n_estimators = 500)
fit = forest1.fit (x_train, y_train)
accuracy = fit.score(x_test,y_test)
predict = fit.predict(x_test)
cmatrix = confusion_matrix (y_test, predict)
print ('Classification Report:',classification_report(y_test,predict))
print ('Accuracy Score', (accuracy_score(y_test,predict)))
print ('Accuracy of Random Forest ', (accuracy))
print ('Confusion Matrix :\n',cmatrix)


# "Finding the feature importances of all the columns in the independent variable with respect to Random Forest Classification Model fitted with resampled variable in above cell for the dimensional reduction process"

# In[31]:


importances = forest1.feature_importances_
indices = np.argsort(importances)[::-1]
print ("Feature Ranking:")
for f in range (x.shape[1]):
    print ("Feature %s (%f)"  %(list (x)[f],importances[indices[f]]))


# "Plotting the Bar Graph to represent the Feature Importances of the Independent variable column and saving the PNG file."

# In[32]:


feat_imp = pd.DataFrame({'Feature': list(x), 'Gini importance': importances[indices]})
plt.rcParams['figure.figsize']= (12,12)
sns.set_style ('whitegrid')
ax= sns.barplot(x ='Gini importance', y = 'Feature', data = feat_imp  )
ax.set (xlabel = 'Gini Importances')
plt.title('Feature Importance')
plt.savefig('Feature Importance Horizontal.png')
plt.show()


# "Plotting the Bar Graph to represent the Feature Importances of the Independent variable column and saving the PNG file."

# In[33]:


pd.Series(forest.feature_importances_,index=x.columns).sort_values(ascending=False).plot(kind='bar',figsize=(10,5))
plt.title('Feature Importance')
plt.savefig('Feature Importance Vertical.png')
plt.show()


# "With respect to feature Importance of the independent variable reducing the dimensions of independent variable for reducing the complexity of model fitting."

# In[34]:


feat_imp.index = feat_imp.Feature
feat_to_keep = feat_imp.iloc[:15].index
display (type(feat_to_keep),feat_to_keep)


# "Passing the Resampled variable after dimensional reduction and Running the function with some appropriate parameters and fitting the Random Forest Classifiers Algorithm and getting the Best Parameters of the algorithm, Accuracy Score, Classification Report and Confusion Matrix between the predicted values and dependent test dataset and also the pickle file with the name Random Forest_resample."

# In[35]:


X_res = pd.DataFrame(x_res)
Y_res = pd.DataFrame(y_res)
X_res.columns = x.columns
param = { 'n_estimators': [100,500,1000,2000]  }
FitModel (X_res [feat_to_keep], Y_res ,'Random Forest_resample',RandomForestClassifier(), param, cv =10)


# "Passing the Resampled variable after dimensional reduction and Running the function with some appropriate parameters and fitting the Support Vector Machine Classifiers Algorithm and getting the Best Parameters of the algorithm, Accuracy Score, Classification Report and Confusion Matrix between the predicted values and dependent test dataset and also the pickle file with the name SVC_resample."

# In[36]:


param = { 'C': [0.1,1,100,1000],
        'gamma':[0.0001,0.001, 0.005, 0.1,1, 3,5,10, 100]
         }
FitModel (X_res [feat_to_keep], Y_res,'SVC_resample',SVC(), param, cv =5)


# "Passing the Resampled variable after dimensional reduction and Running the function with some appropriate parameters and fitting the XGBoost Classifiers Algorithm and getting the Best Parameters of the algorithm, Accuracy Score, Classification Report and Confusion Matrix between the predicted values and dependent test dataset and also the pickle file with the name XGBoost_resample."

# In[37]:


param = { 'n_estimators': [100,111,222,333,444,500,1000,2000]  }
FitModel (X_res [feat_to_keep], Y_res,'XGBoost_resample', XGBClassifier(),param, cv = 5)


# "Loading the pickle file with the algorithm which gives highest accuracy score" 

# In[38]:


model =pickle.load(open("Random Forest","rb"))


# "Predicting the independent variable using the loaded pickle file and getting the accuracy score and best parameters of the loaded pickle file."

# In[39]:


pred1 = model.predict (x_test)
print (accuracy_score (pred1,y_test))
print(model.best_params_)

