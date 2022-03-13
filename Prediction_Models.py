import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier

data = pd.read_csv('Final_Data_5203.csv')

data = data.drop(['imdb_id', 'titleType', 'primaryTitle', 'id', 'original_language'], axis=1)

# RANDOM FOREST

data['rating_bin'] = pd.cut(data['weighted_rating'], 3, labels=["1", "2", "3"])
data['revenue_bin'] = pd.cut(data['revenue'], 5, labels=["1", "2", "3", "4", "5"])
data['popularity_bin'] = pd.cut(data['popularity'], 5, labels=["1", "2", "3", "4", "5"])

# Define X and y
y = data[['rating_bin','revenue_bin', 'popularity_bin']].astype(int)
X = data.drop(['weighted_rating','revenue','revenue_bin','rating_bin', 'Variance', 'popularity_bin', 'popularity'], axis = 1)

# Perform train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model = MultiOutputClassifier(model, n_jobs=-1)

model.fit(X_train,y_train)


acc_forest = model.score(X_test, y_test)
acc_forest = round(acc_forest,5)
acc_forest_print = acc_forest * 100
print('Random Forest Accuracy: {}%'.format(acc_forest_print))
print('Random Forest Accuracy: {}'.format(acc_forest))

# save the model to disk
filename = 'random_forest_model.sav'
#joblib.dump(model, filename)