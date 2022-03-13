import pandas as pd
from io import StringIO
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense


data = pd.read_csv('Final_Data_5203.csv')
print(data.columns.to_list())

print(data.info())

data['weighted_rating'] = data['weighted_rating'].astype(int)
data['runtimeMinutes'] = data['runtimeMinutes'].astype(int)
data['popularity'] = data['popularity'].astype(int)

#data['rating_bin'] = pd.cut(data['weighted_rating'], 3, labels=["1", "2", "3"])
#data['revenue_bin'] = pd.cut(data['revenue'], 5, labels=["1", "2", "3", "4", "5"])

x = data[['runtimeMinutes', 'weighted_rating', 'release_month', 'budget', 'Action', 'Adult', 'Adventure',
          'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family','Fantasy', 'Film-Noir',
          'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Reality-TV', 'Romance', 'Sci-Fi', 'Short',
          'Sport', 'Thriller', 'War', 'Western', 'actor_present', 'directors_present']]

y = data["revenue"]


model = Sequential()
model.add(Dense(12, input_dim=32, activation="relu"))
model.add(Dense(12, activation="relu"))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss='mean_squared_error', optimizer="adam")
print(x.size)
model.fit(x,y, epochs=100, batch_size=1)


# Calling `save('my_model')` creates a SavedModel folder `my_model`.
#model.save("my_model")

# It can be used to reconstruct the model identically.
#reconstructed_model = keras.models.load_model("my_model")


accuracy = model.evaluate(x, y)
print(accuracy)


#predictions = model.predict(x)
