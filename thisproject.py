import pandas as pd
import numpy as np
import scipy.sparse as sp
import copy, requests, re, bs4, os, pickle
from lightfm import LightFM

book_map = pd.read_csv('data/books.csv')
NITEMS = 10000


class DatasetFaster():
    
    '''Used to create dataset of proper format to feed to LightFM'''
    
    def __init__(self):
        pass
    
    def fit(self, n_users, n_items):
        self.n_users = n_users
        self.n_items = n_items
        
    def build_interactions(self, ratings_data, user_ratings=None):
        
        '''
        Creates COO matrices of weights and interactions. 
        Replaces id-0 user with zeroes or custom user_ratings, if provided.
        The model is to be fitted with 0's, then refitted partially with new user_ratings.
        '''
        
        ratings_data_local = copy.deepcopy(ratings_data)
        ratings_data_local['user_id'] = pd.factorize(ratings_data_local['user_id'])[0]
        
        'create interactions and weights'
        interactions = sp.coo_matrix((
            np.repeat(1, ratings_data_local.shape[0]),
            (ratings_data_local['user_id'], ratings_data_local['book_id'] - 1)
        ))
        if interactions.shape[1] < self.n_items:
            interactions = sp.hstack((
                interactions,
                sp.coo_matrix((1, self.n_items - interactions.shape[1])),
            ))
        weights = sp.coo_matrix((
            ratings_data_local['rating'],
            (ratings_data_local['user_id'], ratings_data_local['book_id'] - 1)
        ))
        if weights.shape[1] < self.n_items:
            weights = sp.hstack((
                weights,
                sp.coo_matrix((1, self.n_items - weights.shape[1])),
            ))
            
        'create 0 user vector'
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
        
        'stack 0 user vector and weights/insteractions'
        interactions_upd = sp.vstack((
            user_vector,
            interactions
        ))
        weights_upd = sp.vstack((
            user_vector,
            weights
        ))
        
        return interactions_upd, weights_upd

    
def predict_list(lightfm_model, user_ratings, n=10, escape_series=False):
    
    '''Makes a prediction list from a fitted LightFM model'''
    
    rated = np.array(user_ratings.book_id-1)
    known_positives = book_map.title[rated]
    
    preds = lightfm_model.predict(0, np.arange(NITEMS))
    books_sorted = book_map.title[np.argsort(-preds)]
    
    if escape_series:
        rated_series = book_map.series[rated].unique()
        rated_series = rated_series[rated_series != np.array(None)]
        books_escaped = book_map.title[np.isin(book_map.series, rated_series)]
        return books_sorted[~np.isin(books_sorted, known_positives) & ~np.isin(books_sorted, books_escaped)][:n]
    else:
        return books_sorted[~np.isin(books_sorted, known_positives)][:n]


def fetch_user_ratings_dataset(user_id: int) -> pd.DataFrame:
    
    '''Returns a slice of the original dataset with ratings from a given user, where user_id is set to 0.'''
    
    ratings = pd.read_csv('data/ratings.csv')
    if user_id < 1 or user_id > ratings['user_id'].nunique():
        raise ValueError('User not in dataset')
    df = ratings.loc[ratings['user_id'] == user_id, :]
    df['user_id'] = np.repeat(0, df.shape[0])
    return df


def fetch_user_ratings_goodreads(goodreads_id: int) -> pd.DataFrame:
    
    '''Parses goodreads account with given id and returns a dataset with columns user_id, book_id and rating, 
    where user_id is set to 0.'''
    
    import requests, re, bs4
    from itertools import chain
    
    url = f'https://www.goodreads.com/review/list/{goodreads_id}?print=true'
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 \
    (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    first_page = requests.get(url=url, headers=headers).text
    first_page_bs = bs4.BeautifulSoup(first_page, 'html.parser')
    
    if re.search('Oops - we couldn\'t find that user.', first_page):
        raise ValueError('User does not exist')
    elif re.search('Sorry, that person\'s shelf is private.', first_page):
        raise ValueError('User profile is private')
    
    num_books = int(re.findall('books on Goodreads \((.*) book', first_page)[0])
    
    if num_books == 0:
        raise ValueError('No books rated')
    
    def get_books(page: bs4.BeautifulSoup) -> tuple:
        
        '''
        Yields dict of books from a bs4-formatted page to pass to pd.DataFrame constructor.
        Needed for processing one page separately.
        '''
        
        for book in page.find_all('tr')[2:]:
            title = book.find('td', {'class':'field title'}).a.get('title')
            rating = len(book.find_all('span', {'class': 'staticStar p10'}))
            book_id = book_map.id[book_map.title == title].values

            if len(book_id) and rating:
                yield (0, book_id[0], rating)
    
    def get_pages(num_books: int) -> bs4.BeautifulSoup:
        
        '''Yields bs4-formatted pages passed to get_books'''
        
        for page in range(2, num_books // 20 + (num_books % 20 > 0) + 1):
            url = f"https://www.goodreads.com/review/list/{goodreads_id}?page={page}&print=true"
            html_output = requests.get(url=url, headers=headers).text
            page = bs4.BeautifulSoup(html_output, 'html.parser')
            
            yield from get_books(page)
                 
    df = pd.DataFrame(chain(get_books(first_page_bs), get_pages(num_books)), columns=('user_id', 'book_id', 'rating'))
    
    if df.empty:
        raise ValueError('No books from dataset rated')
    else:
        return df

    
def load_model(model_name):
    with open(str(model_name), 'rb') as f:
        model = pickle.load(f)
    return model
