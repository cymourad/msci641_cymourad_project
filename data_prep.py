import pandas as pd


def remove_corrupt_ids(df):

    corrupt_ids = []
    for i in df['id']:
        try:
            int(i)
        except:
            corrupt_ids.append(i)

    # remove the rows with corrupt ids from the  df
    return df.drop(df[df.id.isin(corrupt_ids)].index)

# load movie_metadata and credits and merge them together


def load_movies_full_df(movies_metadata_path, credits_path, n_votes, desired_columns):
    df_movies = pd.read_csv(movies_metadata_path)
    df_credits = pd.read_csv(credits_path)

    # remove corrupt ids from movis to preapre for merge
    df_movies = remove_corrupt_ids(df_movies)

    # cast movies.id to int for the join to succeed
    df_movies.id = df_movies.id.astype(int)

    # merge/join the 2 dataframes together on the movie id
    df = df_credits.merge(df_movies, on='id')

    # only enough votes
    enough_votes = df[df['vote_count'] >= n_votes]

    # resest index to compensate for dropped rows
    enough_votes.reset_index(drop=True, inplace=True)

    # fill all empty overviews
    enough_votes['overview'] = enough_votes['overview'].fillna('')

    # only keep the desired_columns
    return enough_votes[desired_columns]


def get_index_to_movieId(df):
    indices = pd.Series(df.index, index=df['id']).drop_duplicates()
    return dict(zip(df.index, df.id))


def get_distil_data(distil_data_path):
    print("Loading the distilled data for the bert model ...")
    return pd.read_csv(distil_data_path)
