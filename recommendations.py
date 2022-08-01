from operator import itemgetter

# used for the fast recommendation retreival
# meant to keep memory usage low while still satisfying user's need
MAX_RECOMMENDATIONS_PER_MOVIE = 10

# Takes:
# - movie_id: int
# - cosine_sim: square matrix of size nxn, where n is the number of movies
# - idx2id: dictionary where key is movie index in dataframe and value is the movie id
# - n_rec: number of recommendations to return (default=10)
# Returns:
# - list of length n_rec of the most similar movies


def get_recommendations(movie_id, cosine_sim, idx2id, id2idx, n_rec=10):
    # Get the index of the movie that matches the title
    idx = id2idx[movie_id]

    # Get the pairwsie similarity scores of all movies with that movie
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the n_rec most similar movies
    sim_scores = sim_scores[1:n_rec+1]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top n most similar movies
    return itemgetter(*movie_indices)(idx2id)
    # return popular_movies[['id']].iloc[movie_indices].id.to_list()

# this function takes the cosine_sim matrix and the idx2id dictionary
# and returns a dictionary where:
# - key: movieID
# - val: ordered list of similar movies


def make_recommendations_dict(cosine_sim, idx2id):
    print('Making recommendations dictionary ...')
    rec_dict = {}
    id2idx = {v: k for k, v in idx2id.items()}
    for movieId in idx2id.values():
        rec_dict[movieId] = get_recommendations(
            movieId, cosine_sim, idx2id, id2idx, MAX_RECOMMENDATIONS_PER_MOVIE)
    return rec_dict

# uses pre-computed recommendation matrix to retreive top n recommendations (up to 20)


def get_recommendations_fast(movie_id, rec_dict, n_rec):
    return rec_dict[movie_id][:n_rec]
