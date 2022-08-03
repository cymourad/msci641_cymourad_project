# Bag of words portion is taken form https://gist.github.com/4OH4/f727af7dfc0e6bb0f26d2ea41d89ee55

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
import pandas as pd

# Download stopwords list
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

# Interface lemma tokenizer from nltk with sklearn


class LemmaTokenizer:
    ignore_tokens = [',', '.', ';', ':', '"', '``', "''", '`']

    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc) if t not in self.ignore_tokens]


def get_lemmatized_tfidf_matrix(overviews):
    # Lemmatize the stop words
    tokenizer = LemmaTokenizer()
    token_stop = tokenizer(' '.join(stop_words))

    # Create TF-idf model
    vectorizer = TfidfVectorizer(
        stop_words=token_stop,
        tokenizer=tokenizer,
        lowercase=True,
        max_df=0.95,
        min_df=10)

    # Construct the required TF-IDF matrix by fitting and transforming the data
    return vectorizer.fit_transform(overviews)

# This second section is for bert embeddings

# load the setence embeddings from a csv
# note: each row starts with the movie id, followed by 768 embeddings
# note: the csv does not have a header


def get_bert_embeddings_matrix(path_to_sentence_embeddings):
    df = pd.read_csv(path_to_sentence_embeddings, header=None)
    df.rename(columns={0: 'id'}, inplace=True)
    df.drop_duplicates(subset=['id'], inplace=True)
    return df
