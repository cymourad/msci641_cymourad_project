from recommendations import get_recommendations_fast
import pandas as pd


def load_testing_set(ratings_file_path, accepted_movieIds, neutral_rating, min_n_pos_ratings_per_user):
    print('Loading csv ...')
    df = pd.read_csv('data/IMDB_Ratings/ratings.csv')

    print('Removings reviews for movies that did not have enough ratings ...')
    df = df[df.movieId.isin(accepted_movieIds)]

    print('Removing all neutral and negative ratings ...')
    df = df[df['rating'] > neutral_rating]

    print('Finding active users ...')
    ratings_per_user = df.groupby(
        ['userId'])['userId'].count().reset_index(name='ratingsGiven')

    avid_users = ratings_per_user[ratings_per_user.ratingsGiven >=
                                  min_n_pos_ratings_per_user]

    print('Grouping positive reviews per user ...')
    test_users = df[
        df.userId.isin(avid_users.userId.to_list())
    ].groupby('userId')['movieId'].apply(list).reset_index(name="moviesWatched")

    return test_users


def calculate_x_in_100(recommendations, test_movies):
    overlaps = recommendations.intersection(test_movies)
    x_in_100 = len(overlaps)

    return x_in_100


def get_recommendations_from_baseline(train_movies, rec_dict):
    # how many movies to recommend based on a single watched movie
    n_rec = int(100/len(train_movies))

    recommendations = set()
    for movie_id in train_movies:
        for recId in get_recommendations_fast(movie_id, rec_dict, n_rec):
            recommendations.add(recId)
    return recommendations


def test_recommender(movies_watched, n_train_movies, rec_dict):
    train_movies = movies_watched[:n_train_movies]
    test_movies = movies_watched[n_train_movies:]

    recommendations = get_recommendations_from_baseline(train_movies, rec_dict)

    # calculate x in 100
    return calculate_x_in_100(recommendations, set(test_movies))


def test_recommendations(user_ratings, n_train_movies, rec_dict):
    n_tests = len(user_ratings)
    print(f'Testing on {n_tests} users ...')
    results = []
    for movies_watched in user_ratings:
        res = test_recommender(movies_watched, n_train_movies, rec_dict)
        results.append(res)

    mean = sum(results) / n_tests
    print(f'Mean for {n_tests}: {mean}')

    return mean
