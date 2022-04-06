import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

data = pd.read_csv('Final_Data_5203.csv')
data = data.drop(['imdb_id', 'titleType', 'primaryTitle', 'id', 'original_language'], axis=1)

y = data['weighted_rating'].values
X = data.drop(['weighted_rating', 'revenue', 'Variance'], axis = 1)
X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Weighted Rating

# For Weighted Rating
parameter_space = {
    'n_estimators': [10,50,100],
    'criterion': ['squared_error', 'absolute_error', 'poisson'],
    'max_depth': [10,20,30,40,50],
}

clf = RandomizedSearchCV(RandomForestRegressor(), parameter_space, n_iter=15, cv = 3, scoring = "explained_variance", verbose = True)
clf.fit(X_train,y_train)

clf.best_params_

train_pred = clf.predict(X_train)   # Train predict
test_pred = clf.predict(X_test)     # Test predict

print("For Train:")
print("Mean Square Error is",mean_squared_error(y_train,train_pred)) # Calculating MSE
print("Root Mean Square Error is",mean_squared_error(y_train,train_pred)**(1/2)) # Calculating RMSE
print("Mean Absolute Error is",mean_absolute_error(y_train,train_pred)) # Calculating MAE
print("r2 Score is", r2_score(y_train,train_pred)) # Calculating r2 Score


print("For Test:")
print("Mean Square Error is",mean_squared_error(y_test,test_pred)) # Calculating MSE
print("Root Mean Square Error is",mean_squared_error(y_test,test_pred)**(1/2)) # Calculating RMSE
print("Mean Absolute Error is",mean_absolute_error(y_test,test_pred)) # Calculating MAE
print("r2 Score is", r2_score(y_test,test_pred)) # Calculating r2 Score

#Revenue

# For Revenue
y2 = np.log(data['revenue']).values
X_train, X_test, y_train, y_test = train_test_split(X, y2, test_size=0.2, random_state=42)

clf2 = RandomizedSearchCV(RandomForestRegressor(), parameter_space, n_iter=15, cv = 3, scoring = "explained_variance", verbose = True)
clf2.fit(X_train,y_train)

clf2.best_params_

train_pred2 = clf2.predict(X_train)
test_pred2 = clf2.predict(X_test)

print("For Train:")
print("Mean Square Error is",mean_squared_error(y_train,train_pred2)) # Calculating MSE
print("Root Mean Square Error is",mean_squared_error(y_train,train_pred2)**(1/2)) # Calculating RMSE
print("Mean Absolute Error is",mean_absolute_error(y_train,train_pred2)) # Calculating MAE
print("r2 Score is", r2_score(y_train,train_pred2)) # Calculating r2 Score

print("For Test:")
print("Mean Square Error is",mean_squared_error(y_test,test_pred2)) # Calculating MSE
print("Root Mean Square Error is",mean_squared_error(y_test,test_pred2)**(1/2)) # Calculating RMSE
print("Mean Absolute Error is",mean_absolute_error(y_test,test_pred2)) # Calculating MAE
print("r2 Score is", r2_score(y_test,test_pred2)) # Calculating r2 Score

# save the model to disk
filename = 'random_forest_model.sav'
#joblib.dump(clf2, filename)