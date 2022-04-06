import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

data = pd.read_csv('Final_Data_5203.csv')
data = data.drop(['imdb_id', 'titleType', 'primaryTitle', 'id', 'original_language'], axis=1)

# Define X and y
y = data['weighted_rating'].values
y2 = np.log(data['revenue']).values
X = data.drop(['weighted_rating', 'revenue', 'Variance'], axis = 1)
X = StandardScaler().fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=2)


print(X_train.shape)
print(y_train.shape)


# Splitting the data into training and testing data

#Predicting Rating
regr = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1)
clf = cross_val_score(regr, X_train, y_train,scoring='neg_mean_squared_error', cv=10) # 10 Fold cross validation
print("Cross-Validation Score:",np.mean(clf))
regr.fit(X_train, y_train)

print(regr.score(X_test, y_test))

y_pred = regr.predict(X_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Coefficients:', regr.coef_)
print('Intercept:', regr.intercept_)


#Predicting Revenue
X_train, X_test, y_train, y_test = train_test_split(X, y2, test_size=0.2, random_state=42)
regr2 = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1)
clf2 = cross_val_score(regr2, X_train, y_train,scoring='neg_mean_squared_error', cv=10) # 10 Fold cross validation
print("Cross-Validation Score:",np.mean(clf2))
regr2.fit(X_train, y_train)

print(regr2.score(X_test, y_test))

y_pred = regr2.predict(X_test)

print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
print('Coefficients:', regr2.coef_)
print('Intercept:', regr2.intercept_)