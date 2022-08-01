# computer the cosine similarity on the bag of words tfidf and the features count vectors
from scipy.sparse import hstack
from sklearn.metrics.pairwise import cosine_similarity

FEATURE_WEIGHT = 1
BOW_WEIGHT = 1


def get_cosine_sim_matrix(overview_matrix, features_matrix,
                          overview_weight=BOW_WEIGHT, feature_weight=FEATURE_WEIGHT):
    full_input_matrix = hstack(
        (overview_weight*overview_matrix, feature_weight*features_matrix))

    return cosine_similarity(full_input_matrix, full_input_matrix)
