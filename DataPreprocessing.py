import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import re
from sklearn.ensemble import RandomForestClassifier
from IPython.display import display
import statsmodels.api as sm
from sklearn.metrics import accuracy_score

# reading two csv files
data1 = pd.read_csv('DatasetUsed\Title_Basis.csv', low_memory=False)
data2 = pd.read_csv('DatasetUsed\Title_Rating.csv', low_memory=False)
data3 = pd.read_csv('DatasetUsed\Budget and Revanue.csv', low_memory=False)
data4 = pd.read_csv('DatasetUsed\credits.csv', low_memory=False)

data1.rename(columns={'tconst': 'imdb_id'}, inplace=True)
data2.rename(columns={'tconst': 'imdb_id'}, inplace=True)
#data3.rename(columns={'imdb_id': 'id'}, inplace=True)

# using merge function by setting how='inner'
output1 = pd.merge(data1, data2,
                   on='imdb_id',
                   how='inner')

output2 = pd.merge(output1, data3,
                   on='imdb_id',
                   how='inner')


output_final = pd.merge(output2, data4,
                        on='id',
                        how='inner')


# displaying result
dataset = output_final.drop(columns=['originalTitle', 'isAdult', 'endYear', 'original_title', 'overview', 'production_companies', 'production_countries',
                                     'startYear', 'runtime', 'genres_y', 'spoken_languages', 'status', 'tagline', 'vote_average', 'vote_count', 'title', 'production_companies_number', 'production_countries_number', 'spoken_languages_number'])


#Drop all non-movies
dataset = dataset[(dataset['titleType'] == 'movie') | (dataset['titleType'] == 'tvMovie')]


# EXTRACTING RELEASE MONTH
dataset = dataset[dataset['release_date'].notna()]

#print(dataset.describe())
#print(dataset['release_date'].value_counts())

release_date_split = dataset['release_date'].astype(str)
date_list = []
for date in release_date_split:
    date_splitted = date.split('/')
    if date_splitted[1][0] == '0':
        month = date_splitted[1][1]
        date_list.append(month)
    else:
        date_list.append(date_splitted[1])

dataset['release_month'] = date_list

# HANDLING GENRES

#store genres
dataset = dataset[dataset['genres_x'] != '\\N']

# Format the genre_x column

test_to_split = dataset['genres_x']
list_arrays = []
for test in test_to_split:
    test_splitted = test.split(',')
    line_list = []
    for t in test_splitted:
        line_list.append(t)
    list_arrays.append(line_list)

dataset['genre'] = list_arrays

# Gather list of genres
genre_list = []

for movie_genre in dataset['genre']:
    for genre in movie_genre:
        if genre not in genre_list:
            genre_list.append(genre)

#print(genre_list)   Printing all the genres

# Perform one-hot encoding for genres to make it more suitable for the models

from sklearn.preprocessing import MultiLabelBinarizer

mlb = MultiLabelBinarizer()
dataset = dataset.join(pd.DataFrame(mlb.fit_transform(dataset.pop('genre')),
                                    columns=mlb.classes_,
                                    index=dataset.index))

pd.set_option("display.max_rows", None, "display.max_columns", None)



#Weigted Voting Rating  (IMDb Formula)

"""
(WR) = (v ÷ (v + m)) × R + (m ÷ (v + m)) × C
WR: weighted rating
R: average for the movie (mean) = (Rating)
v: number of votes for the movie = (votes)
m: minimum votes required to be listed in the Top 250
C: the mean vote across the whole report
"""

votes_num = dataset['numVotes']
vote_num_avg = dataset['numVotes'].mean()
mov_rating = dataset['averageRating']
rate_mean = dataset['averageRating'].mean()

dataset['weighted_rating'] = (((votes_num / (votes_num + vote_num_avg)) * mov_rating) + (vote_num_avg / (votes_num + vote_num_avg)) * rate_mean)

"""
# Calculate the budget variance (VARIANCE FORMULA)
budgeted_variance = dataset['revenue'].astype(int) - dataset['budget'].astype(int)
dataset['Variance'] = budgeted_variance
"""

#Extract the main Directors of the movie

def format_director(cast_series):
    cast_total = []

    # Only search for "'name':" and discard everything else
    director_search = re.compile(r"'department': 'Directing'(.)*'Director', 'name': [A-Z \'\"]+,?", re.I)
    # (.)* searches for 0 or more of any character
    # regex string searches for "department: Directing .... 'Director', 'name': ....," (ending with the comma)
    # this is because there are sound/art directors etc. and we only want the main directors

    for movie in cast_series:
        cast_list = movie.split('}')
        cast_per_movie = []
        for cast in cast_list:
            if (len(cast) <= 2):  # to factor for the last entry in movie.split(), which will be ']'
                continue
            else:
                try:
                    cast_name = re.search(director_search, cast)
                    cast_name = re.sub(".*'name'", '', cast_name[0])  # removes all characters before the name
                    cast_name = re.sub("[:,\']", '',
                                       cast_name)  # these 3 rows remove special characters and leading and trailing spaces
                    cast_name = cast_name.strip()
                    cast_name = re.sub("^\'|\'$", "", cast_name)
                    cast_per_movie.append(cast_name)
                except:
                    continue

        cast_total.append(cast_per_movie)
    return cast_total


#Extract the main actors of the movie

def format_actors(crew_series):
    crew_total = []

    # Only search for "'name':" and discard everything else
    actors_search = re.compile(r"'name': [A-Z \'\"]+,?", re.I)
    # (.)* searches for 0 or more of any character
    # regex string searches for "department: Directing .... 'Director', 'name': ....," (ending with the comma)
    # this is because there are sound/art directors etc. and we only want the main directors

    for movie in crew_series:
        crew_list = movie.split('}')
        crew_per_movie = []
        for crew in crew_list:
            if (len(crew) <= 2):  # to factor for the last entry in movie.split(), which will be ']'
                continue
            else:
                try:
                    crew_name = re.search(actors_search, crew)
                    crew_name = re.sub(".*'name'", '', crew_name[0])  # removes all characters before the name
                    crew_name = re.sub("[:,\']", '',
                                       crew_name)  # these 3 rows remove special characters and leading and trailing spaces
                    crew_name = crew_name.strip()
                    crew_name = re.sub("^\'|\'$", "", crew_name)
                    crew_per_movie.append(crew_name)
                except:
                    continue

        crew_total.append(crew_per_movie)
    return crew_total

list_of_actors = format_actors(dataset['cast'])

# HANDLE THE BEST ACTORS

data_best_directors = pd.read_csv('DatasetUsed\Best Actors - Top 250.csv')
dataset_best_actors = data_best_directors[['Position', 'Name']]


director_present_count = []
list_dir_names = dataset_best_actors['Name'].tolist()
for actor_array in list_of_actors:
    count = 0
    count_per_movie = []
    for actor in actor_array:
        if actor in list_dir_names:
            count = count + 1
    if count != 0:
        director_present_count.append(count)
    else:
        director_present_count.append(0)


dataset['actor_present'] = director_present_count
#dataset['actors_array'] = format_actors(dataset['cast'])

# HANDLE THE BEST DIRECTORS
data_best_directors = pd.read_csv('DatasetUsed\Top_250_directors_IMDb.csv')
data_best_directors = data_best_directors[['Position', 'Name']]
list_of_dirs = format_director(dataset['crew'])

director_present_count = []
list_dir_names = dataset_best_actors['Name'].tolist()
for dir_array in list_of_dirs:
    count = 0
    count_per_movie = []
    for direc in dir_array:
        if direc in list_dir_names:
            count = count + 1
    if count != 0:
        director_present_count.append(count)
    else:
        director_present_count.append(0)

dataset['directors_present'] = director_present_count

#Prepare to Export
dataset.drop(['genres_x', 'averageRating', 'numVotes', 'release_date', 'cast', 'crew'], axis=1, inplace=True)

# Handle Runntime minutes
dataset.loc[dataset["runtimeMinutes"] == "\\N", "runtimeMinutes"] = "0"
dataset['runtimeMinutes'] = dataset['runtimeMinutes'].astype(float, errors = 'raise')
runtime_mean = round(dataset['runtimeMinutes'].mean(), 2)
dataset['runtimeMinutes'] = dataset['runtimeMinutes'].replace(0.0, runtime_mean)

# Convert from object to int/float
dataset['popularity'] = dataset['popularity'].astype(float, errors = 'raise')
dataset['release_month'] = dataset['release_month'].astype(int, errors = 'raise')


# BUDGET AND REVANUE VALUE CLEANING

# REPLACE 0 VALUES WITH MEAN()

dataset_mean = dataset.copy()
dataset_droped_null = dataset_mean[dataset_mean['budget'] != 0]
budget_mean = dataset_droped_null['budget'].mean()
budget_mean = round(budget_mean,2)
dataset_mean['budget'] = dataset_mean['budget'].mask(dataset_mean['budget'] == 0).fillna(budget_mean)

dataset_droped_null = dataset_mean[dataset_mean['revenue'] != 0]
revanue_mean = dataset_droped_null['revenue'].mean()
revanue_mean = round(revanue_mean,2)
dataset_mean['revenue'] = dataset_mean['revenue'].mask(dataset_mean['revenue'] == 0).fillna(revanue_mean)

# Calculate the budget variance (VARIANCE FORMULA)
budgeted_variance = dataset_mean['revenue'].astype(int) - dataset_mean['budget'].astype(int)
dataset_mean['Variance'] = budgeted_variance

#dataset_mean.to_csv(r'C:\Users\USER\PycharmProjects\pythonProject\Final_Data_All_Mean.csv', index=False)

# DELETE 0 VALUES

dataset_null = dataset.copy()
dataset_null = dataset_null[dataset_null['budget'] != 0]
dataset_null = dataset_null[dataset_null['revenue'] != 0]

# Calculate the budget variance (VARIANCE FORMULA)
budgeted_variance = dataset_null['revenue'].astype(int) - dataset_null['budget'].astype(int)
dataset_null['Variance'] = budgeted_variance

#dataset_null.to_csv(r'C:\Users\USER\PycharmProjects\pythonProject\Final_Data_5203.csv', index=False)

