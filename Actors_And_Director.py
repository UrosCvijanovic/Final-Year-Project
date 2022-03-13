import pandas as pd

data = pd.read_csv('Cleaned_Metadata.csv')
data_best_actors = pd.read_csv('Best Actors - Top 250.csv')

#Extracting the features
dataset_best_actors = data_best_actors[['Position', 'Name']]
dataset_staff = data[['imdb_id', 'director', 'actors_array']]
#print(dataset_staff.columns.to_list())
#print(dataset_best_actors.columns.to_list())

list_names = dataset_best_actors['Name'].tolist()
Dict = dataset_best_actors.set_index('Name').to_dict()['Position']
print(list_names)






#dataset_best_actors = dataset_best_actors.reset_index()  # make sure indexes pair with number of rows
#for data in dataset_staff['actors_array']: