# movie-ticket
import numpy as np 
import pandas as pd 
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('tmdb_5000_credits.csv')
movies.head(1)
credits.head(1)
movies = movies.merge(credits,on='title')
movies.shape
movies.head()
movies.info()
movies = movies[['title','overview','genres','keywords']]
movies.head()
movies.isnull().sum()
movies.dropna(inplace=True)
movies.duplicated().sum()
credits.head()

movies = movies.merge(credits,on='title')
movies.head()
movies = movies[['movie_id','title','overview','genres','keywords','cast','crew']]
movies.head()
movies.isnull().sum()

movies.iloc[0].genres
import ast
def convert(obj):
 L=[]
 for i in ast.literal_eval(obj):
  L.append(i['name'])
 return L
movies['genres'] = movies['genres'].apply(convert)
movies.head()
movies['keywords'] = movies['keywords'].apply(convert)
movies.head()
def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter+=1
    return L
movies['cast'] = movies['cast'].apply(convert)
movies.head()
def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L
movies['crew'] = movies['crew'].apply(fetch_director)
movies.head()
movies['overview'] = movies['overview'].apply(lambda x:x.split())
movies.head()
movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","")for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","")for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","")for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","")for i in x])
movies.head()
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
movies.head()
new_df=movies[['movie_id','title','tags']]
new_df
new_df['tags']=new_df['tags'].apply(lambda x: " ".join(x))
new_df.head()
new_df['tags'][0]
new_df['tags']=new_df['tags'].apply(lambda x:x.lower())
new_df.head()
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')
vector = cv.fit_transform(new_df['tags']).toarray()
vector.shape
vector[0]
cv.get_feature_names()
from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vector)
similarity.shape
def recommend(movie):
    index = new_df[new_df['title'] == movie].index[0]
    distances = sorted(list(enumerate(similarity[index])),reverse=True,key = lambda x: x[1])
    for i in distances[1:6]:
        print(new_df.iloc[i[0]].title)
recommend('Batman Begins')
import pickle
pickle.dump(new_df,open('movies.pkl','wb'))
new_df
pickle.dump(new_df.to_dict(),open('movie_dict.pkl','wb'))
pickle.dump(similarity,open('similarity.pkl','wb'))
