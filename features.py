from ast import literal_eval
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer


def parse_into_python_objects(df, features):
    print('Parsing stringified objects into Python readable objects ...')
    for feature in features:
        df[feature] = df[feature].apply(literal_eval)
    return df

# Get the director's name from the crew feature. If director is not listed, return NaN


def get_director(x):
    for i in x:
        if i['job'] == 'Director':
            return i['name']
    return np.nan

# Returns the list top n elements or entire list; whichever is more.


def get_list(x, top_n):
    if isinstance(x, list):
        names = [i['name'] for i in x]
        # Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
        if len(names) > top_n:
            names = names[:top_n]
        return names

    # Return empty list in case of missing/malformed data
    return []


def get_director_from_crew(crew):
    print('Getting director names ...')
    return crew.apply(get_director)


def get_top_n_per_feature(df, feature_and_n):
    for feature, top_n in feature_and_n:
        print(f'Extracting top {top_n} {feature} ...')
        df[feature] = df[feature].apply(get_list, args=(top_n,))
    return df

# Function to convert all strings to lower case and strip names of spaces


def clean_data(x):
    if isinstance(x, list):
        return [str.lower(i.replace(" ", "")) for i in x]
    else:
        # Check if director exists. If not, return empty string
        if isinstance(x, str):
            return str.lower(x.replace(" ", ""))
        else:
            return ''

# join all the features together into one string thacan be processed by a count-vectorizer


def join_features(x, features=['cast', 'genres', 'director', 'production_companies']):
    joined_features = ''
    for feature in features:
        clean_x = clean_data(x[feature])
        joined_features += ' '.join(clean_x) + ' '
    return joined_features


def get_features(df, features_names, feature_and_n, features_in_soup=['cast', 'genres', 'director', 'production_companies'], make_soup=True):
    print('Preparing features ...')

    df = parse_into_python_objects(df, features_names)
    df['director'] = get_director_from_crew(df['crew'])
    df = get_top_n_per_feature(df, feature_and_n)

    if make_soup:
        return df.apply(join_features, args=(features_in_soup,), axis=1)
    else:
        return df


def make_soup(df, features_in_soup=['cast', 'genres', 'director', 'production_companies']):
    return df.apply(join_features, args=(features_in_soup,), axis=1)


def get_feature_count_matrix(features):
    cv = CountVectorizer()
    return cv.fit_transform(features)
