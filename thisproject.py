import pandas as pd
import numpy as np
import scipy.sparse as sp
import copy
import requests
import re
import bs4
from surprise import Reader, Dataset
from surprise import KNNWithMeans, SVD, SVDpp
from lightfm import LightFM
# from lightfm.data import Dataset as LightfmDataset

ratings = pd.read_csv('data/ratings.csv')
book_map = pd.read_csv('data/books.csv')


class LightFM_Recommender():
    def __init__(self):
        pass

    def build_interactions(self, ratings_data):
        interactions = sp.coo_matrix((
            np.repeat(1, ratings_data.shape[0]),
            (ratings_data['user_id'], ratings_data['book_id'] - 1)
        ))
        if interactions.shape[1] < self.n_items:
            interactions = sp.hstack((
                interactions,
                sp.coo_matrix((1, self.n_items - interactions.shape[1])),
            ))
        weights = sp.coo_matrix((
            ratings_data['rating'],
            (ratings_data['user_id'], ratings_data['book_id'] - 1)
        ))
        if weights.shape[1] < self.n_items:
            weights = sp.hstack((
                weights,
                sp.coo_matrix((1, self.n_items - weights.shape[1])),
            ))

        return interactions, weights

    def update_interacions(self, user_ratings=None):
        if user_ratings is not None:
            user_vector = sp.coo_matrix((
                np.repeat(1, user_ratings.shape[0]),
                (user_ratings['user_id'], user_ratings['book_id'] - 1)
            ))
            if user_vector.shape[1] < self.n_items:
                user_vector = sp.hstack((
                    user_vector,
                    sp.coo_matrix((1, self.n_items - user_vector.shape[1])),
                ))
        else:
            user_vector = sp.coo_matrix((1, self.n_items))

        interactions_upd = sp.vstack((
            user_vector,
            self.interactions
        ))
        weights_upd = sp.vstack((
            user_vector,
            self.weights
        ))
        return interactions_upd, weights_upd

#    def build_item_features(self, item_features):
#        # from sklearn.preprocessing import normalize
#        feats_kostyl = sp.hstack((
#            item_features,
#            sp.coo_matrix((item_features.shape[0], item_features.shape[0] - item_features.shape[1])),
#        ))
#        # return sp.identity(item_features.shape[0])+feats_kostyl
#        # return feats_kostyl
#        # return normalize(feats_kostyl, norm='l1', axis=1)
#        return feats_kostyl

    def fit(self, ratings_data, algorithm, user_feats=None, item_feats=None):
        self.n_users = ratings_data['user_id'].nunique()
        self.n_items = 10000
        ratings_data_local = copy.deepcopy(ratings_data)
        # reset user id
        if ratings_data_local.shape[0] < 5976479:
            ratings_data_local['user_id'] = ratings_data_local['user_id'].replace(
                dict(zip(ratings_data_local['user_id'].unique(), np.arange(self.n_users))))
        #
        self.algorithm = copy.deepcopy(algorithm)
        self.interactions, self.weights = self.build_interactions(ratings_data_local)
        if item_feats is None:
            self.item_features = None
        elif item_feats.shape[0] == item_feats.shape[1] == self.n_items:
            self.item_features = item_feats
        else:
            raise ValueError

        interactions_upd, weights_upd = self.update_interacions()

        self.algorithm.fit(
            interactions=interactions_upd,
            # user_features = self.user_features,
            item_features=self.item_features,
            sample_weight=weights_upd,
            epochs=10
        )
        return self

    def predict_list(self, user_ratings, n=10, handle_series=False):
        interactions_new, weights_new = self.update_interacions(user_ratings)

        algorithm_local = copy.deepcopy(self.algorithm)
        algorithm_local.fit_partial(
            interactions=interactions_new,
            # user_features = self.user_features,
            item_features=self.item_features,
            sample_weight=weights_new
        )
        algorithm_local.item_biases = np.zeros_like(algorithm_local.item_biases)

        # titles = pd.read_csv('data/books.csv')['title'].values
        titles = book_map['title'].values
        known_positives = titles[weights_new.tocsr()[0].indices]
        preds = algorithm_local.predict(0, np.arange(self.n_items))
        books_sorted = titles[np.argsort(-preds)]

        return books_sorted[~np.isin(books_sorted, known_positives)][:n]


def recommend_list(user_ratings, ratings_data, algorithm, verbose=False, remove_rated=True, n=10):
    reader = Reader(rating_scale=(1, 5))
    data_full = Dataset.load_from_df(
        user_ratings.copy().append(ratings_data), reader
    ).build_full_trainset()
    algorithm.fit(data_full)
    preds = [algorithm.predict(0, i).est for i in data_full['book_id'].unique()]

    titles = book_map['title'].values
    books_sorted = titles[np.argsort(-preds)]
    known_positives = titles[user_ratings['book_id'].unique() - 1]

    if remove_rated:
        return books_sorted[~np.isin(books_sorted, known_positives)][:n]
    else:
        return books_sorted[:n]


def fetch_user_ratings_dataset(user_id):
    ratings = pd.read_csv('data/ratings.csv')
    if user_id < 1 or user_id > ratings['user_id'].nunique():
        raise ValueError
    df = ratings.loc[ratings['user_id'] == user_id, :]
    df['user_id'] = np.repeat(0, df.shape[0])
    return df


def fetch_user_ratings_goodreads(goodreads_id):
    import requests
    url = f"https://www.goodreads.com/review/list/{goodreads_id}?print=true"
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 \
    (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    html_output = requests.get(url=url, headers=headers).text

    import re
    if re.search('</a>\n        Oops - we couldn\'t find that user.\n      </div>', html_output):
        raise ValueError('User does not exist')

    num_books = int(re.findall('books on Goodreads \((.*) books\)', html_output)[0])
    for p in np.arange(2, (num_books // 20 + (num_books % 20 > 0) + 1)):
        url = f"https://www.goodreads.com/review/list/{goodreads_id}?page={p}&print=true"
        html_output += requests.get(url=url, headers=headers).text

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html_output, 'html.parser')

    bad_stuff_titles = ['Goodreads Home', None, 'My group discussions', 'Messages', 'Friends',
                        'did not like it', 'it was ok', 'liked it', 'really liked it', 'it was amazing', ]
    books = [i.get('title') for i in soup.find_all('a')
             if i.get('title') not in bad_stuff_titles]

    stars = [i.get('class')[1] for i in soup.find_all('span')
             if i.get('class') in [['staticStar', 'p0'], ['staticStar', 'p10']]]

    def groupwise(iterable):
        a = iter(iterable)
        return zip(a, a, a, a, a)

    rates = [(s1, s2, s3, s4, s5).count('p10') for (s1, s2, s3, s4, s5) in groupwise(stars)]

    # book_map = pd.read_csv('data/books.csv')[['id', 'title', 'authors']]

    # books = books[np.isin(
    # df = pd.DataFrame({
    #    'user_id': 0,
    #    'book_id': book_map['id'][book_map['title'].isin(books)].values,
    #    'rating': rates
    # })
    df = pd.merge(pd.DataFrame({'title': books, 'rating': rates}), book_map[['id', 'title']], on='title')
    df = df.loc[df['rating'] != 0].drop('title', axis=1).rename({'id': 'book_id'}, axis=1)
    df['user_id'] = np.repeat(0, df.shape[0])
    df = df.reindex(columns=['user_id', 'book_id', 'rating'])

    if df.empty:
        raise ValueError('No matching books rated')
    else:
        return df
