from data_prep import load_movies_full_df, get_index_to_movieId
from bow import get_lemmatized_tfidf_matrix
from features import get_features, get_feature_count_matrix
from similarity import get_cosine_sim_matrix
from test import load_testing_set, test_recommendations
from recommendations import make_recommendations_dict

MIN_VOTES_PER_MOVIE = 50
NEUTRAL_RATING = 2.5
MIN_POSITIVE_VOTES_PER_USER = 20
DESIRED_COLUMNS = ['id', 'cast', 'title', 'crew',
                   'genres', 'overview', 'production_companies']


def append_to_analytics(mean_x_in_100, weight_features, weight_bow, n_train_movies):
    with open('analytics.csv', 'a') as f:
        f.write(
            f'{mean_x_in_100}, {weight_features}, {weight_bow}, {n_train_movies}\n')


def main():
    # get the basic dataframe
    # that has the desired features and that only keeps movies that are frequently watched (i.e. rated)
    df = load_movies_full_df(
        movies_metadata_path='data/IMDB_Ratings/movies_metadata.csv',
        credits_path='data/IMDB_Ratings/credits.csv',
        n_votes=MIN_VOTES_PER_MOVIE,
        desired_columns=DESIRED_COLUMNS)

    # get the tf-idf matrix of the overviews of the movies
    # used by the base model
    lemmatized_tfidf_matrix = get_lemmatized_tfidf_matrix(df['overview'])

    # get the features of a movie to augment its data with it
    df['features'] = get_features(df, ['cast', 'crew', 'genres', 'production_companies'], [
                                  ('cast', 5), ('genres', 3), ('production_companies', 2)])

    # get a count-vectorized matrix of the features
    features_count_matrix = get_feature_count_matrix(df['features'])

    # TESTING

    print('\n>> TESTING')

    test_df = load_testing_set(
        ratings_file_path='data/IMDB_Ratings/ratings.csv',
        accepted_movieIds=df['id'].to_list(),
        neutral_rating=NEUTRAL_RATING,
        min_n_pos_ratings_per_user=MIN_POSITIVE_VOTES_PER_USER
    )

    # get recommendation matrix for fast testing
    idx2id = get_index_to_movieId(df)

    for overview_weight in [1]:  # [0, 0.5, 1, 2, 4]:
        for feature_weight in [1]:  # [4, 2, 1, 0.5, 0]:
            for n_train_movies in [10]:  # [10, 5]:

                print(
                    f'\n >> Testing with {n_train_movies} training movies, {overview_weight} overview weight, and {feature_weight} feature weight.')

                # construct a similarity matrix
                sim_mx = get_cosine_sim_matrix(
                    lemmatized_tfidf_matrix, features_count_matrix, overview_weight, feature_weight)

                rec_dict = make_recommendations_dict(sim_mx, idx2id)

                # test the recommendations on the test users
                mean_x_in_100 = test_recommendations(
                    user_ratings=test_df['moviesWatched'].to_list(),
                    n_train_movies=n_train_movies,
                    rec_dict=rec_dict
                )

            # append_to_analytics(
            #     mean_x_in_100, feature_weight, overview_weight, n_train_movies)


if __name__ == "__main__":
    main()
