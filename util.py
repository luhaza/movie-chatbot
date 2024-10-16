"""
Utility methods to load movie data from data files.
Also includes recommendation modules.

You do not need to modify anything in this file.   
"""
import csv
from typing import Tuple, List, Dict
import numpy as np
import pickle


def load_ratings(src_filename, delimiter: str = '%',
                 header: bool = False) -> Tuple[List, np.ndarray]:
    title_list = load_titles('data/movies.txt')
    user_id_set = set()
    with open(src_filename, 'r') as f:
        content = f.readlines()
        for line in content:
            user_id = int(line.split(delimiter)[0])
            if user_id not in user_id_set:
                user_id_set.add(user_id)
    num_users = len(user_id_set)
    num_movies = len(title_list)
    mat = np.zeros((num_movies, num_users))

    with open(src_filename) as f:
        reader = csv.reader(f, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL)
        if header:
            next(reader)
        for line in reader:
            mat[int(line[1])][int(line[0])] = float(line[2])
    return title_list, mat


def load_titles(src_filename: str, delimiter: str = '%',
                header: bool = False) -> List:
    title_list = []
    with open(src_filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL)
        if header:
            next(reader)

        for line in reader:
            movieID, title, genres = int(line[0]), line[1], line[2]
            if title[0] == '"' and title[-1] == '"':
                title = title[1:-1]
            title_list.append([title, genres])
    return title_list


def load_sentiment_dictionary(src_filename: str, delimiter: str = ',',
                              header: bool = False) -> Dict:
    with open(src_filename, 'r') as f:
        reader = csv.reader(f, delimiter=delimiter, quoting=csv.QUOTE_MINIMAL)
        if header:
            next(reader)
        return dict(reader)

def similarity(u, v):
    """Calculates the cosine similarity between two vectors.

    Assume that the two arguments have the same shape.

    :param u: one vector, as a 1D numpy array
    :param v: another vector, as a 1D numpy array

    :returns: the cosine similarity between the two vectors
    """
    if (np.linalg.norm(u) * np.linalg.norm(v)) == 0:
        return 0
    else:
        similarity = np.dot(u,v) / (np.linalg.norm(u) * np.linalg.norm(v))
        return similarity

def binarize(ratings: np.ndarray, threshold=2.5):
    """Return a binarized version of the given matrix.

    To binarize a matrix, replace all entries above the threshold with 1.
    and replace all entries at or below the threshold with a -1.

    Entries whose values are 0 represent null values and should remain at 0.

    Note that this method is intentionally made static, as you shouldn't use
    any attributes of Chatbot like self.ratings in this method.

    :param ratings: a (num_movies x num_users) matrix of user ratings, from
     0.5 to 5.0
    :param threshold: Numerical rating above which ratings are considered
    positive

    :returns: a binarized version of the movie-rating matrix
    """
    binarized_ratings = np.vectorize(lambda x: (1.0 if x > threshold else -1.0) if x != 0 else 0)(ratings)

    return binarized_ratings

def recommend(user_rating_all_movies: np.ndarray, ratings_matrix:np.ndarray, num_return: int=10) -> list:
    """Generates a list of indices of movies to recommend using collaborative filtering.
    
    To read more about collaborative filtering, see the following: 
    - Jure Leskovec, Anand Rajaraman, Jeff Ullman. 2014. Mining of Massive Datasets. 
    Chapter 9 3rd edition. pages 319-339
    - https://web.stanford.edu/class/cs124/lec/collaborativefiltering21.pdf
    
    Other notes: 
        - As a precondition, user_ratings and ratings_matrix are both binarized.
        - This function excludes movies the user has already rated
    
    Arguments: 
    - user_rating_all_movies (np.ndarray): a binarized 1D numpy array of the user's rating for each
        movie in ratings_matrix
    - ratings_matrix (np.ndarray): a binarized 2D numpy matrix of all ratings, where
      `ratings_matrix[i, j]` is the rating for movie i by user j
    - num_return (int): the number of recommendations to generate

    Returns:
        - a list of num_return movie indices corresponding to movies in ratings_matrix,
      in descending order of recommendation
    """
    # binarize ratings
    ratings_matrix = binarize(ratings_matrix, threshold=2.5)

    # Populate this list with k movie indices to recommend to the user.
    recommendations = []

    nonzero_indices = [i for i in range(len(user_rating_all_movies)) if user_rating_all_movies[i] != 0]

    ratings = {}
    for i in range(np.asarray(ratings_matrix).shape[0]):
        rating = 0
        for j in nonzero_indices:
            rating += similarity(ratings_matrix[i], ratings_matrix[j]) * user_rating_all_movies[j]
        ratings[i] = rating
    sorted_ratings = sorted(ratings.items(), key=lambda kv: kv[1], reverse=True)

    for movie,rating in sorted_ratings:
        if len(recommendations) >= num_return:
            break
        if user_rating_all_movies[movie] == 0:
            recommendations.append(movie)

    return recommendations

def load_rotten_tomatoes_dataset(): 
    with open("data/rotten_tomatoes.pkl", "rb") as f:
        texts, y = pickle.load(f)
    assert len(texts) == len(y) == 10000 # 10000 entries 
    return texts, y
