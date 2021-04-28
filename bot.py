import pandas as pd
import numpy as np
import scipy.sparse as sp
import time
import copy
import os
import pickle
from surprise import Reader
from surprise import Dataset as SurpriseDataset
from surprise import KNNWithMeans, SVD, SVDpp
from lightfm import LightFM
from lightfm.data import Dataset as LightfmDataset
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
pd.options.mode.chained_assignment = None

ratings = pd.read_csv('data/ratings.csv')
#ratings_random = pd.read_csv('data/ratings_random.csv')
book_map = pd.read_csv('data/books.csv')[['id', 'title', 'authors']]

def recommend_list(user_ratings, ratings_data, algorithm, verbose = False, remove_rated = True):
    reader = Reader(rating_scale=(1, 5))
    data_full = SurpriseDataset.load_from_df(user_ratings.append(ratings_data), reader).build_full_trainset()
    
    algorithm.fit(data_full)
    
    preds = [algorithm.predict(0, i).est for i in data_full['book_id'].unique()]
    #for i in data_full['book_id'].unique():
    #    preds.append(algorithm.predict(user_ratings.user_id.unique()[0], i).est)
    
    recs = pd.DataFrame({'book_id' : data_full['book_id'].unique(), 'estimated_rating' : preds})
    
    if remove_rated:
        recs = recs.loc[~recs['book_id'].isin(user_ratings['book_id'])]
    recs = recs.sort_values('estimated_rating', ascending = False).head(10)
    
    book_map = pd.read_csv('data/books.csv')[['id', 'title', 'authors']]
    
    return [book_map.loc[i-1, 'title'] for i in recs['book_id']]

class LightFM_Recommender():
    def __init__(self):
        pass
    
    def build_interactions(self, ratings_data):
        interactions = sp.coo_matrix((
            np.repeat(1, ratings_data.shape[0]),
            (ratings_data['user_id'], ratings_data['book_id']-1)
        ))
        if interactions.shape[1]<self.n_items:
            interactions = sp.hstack((
                interactions,
                sp.coo_matrix((1, self.n_items-interactions.shape[1])),
            ))
        weights = sp.coo_matrix((
            ratings_data['rating'],
            (ratings_data['user_id'], ratings_data['book_id']-1)
        ))
        if weights.shape[1]<self.n_items:
            weights = sp.hstack((
                weights,
                sp.coo_matrix((1, self.n_items-weights.shape[1])),
            ))
        
        return interactions, weights
    
    def update_interacions(self, user_ratings=None):
        if user_ratings is not None:
            user_vector = sp.coo_matrix((
                np.repeat(1, user_ratings.shape[0]),
                (user_ratings['user_id'], user_ratings['book_id']-1)
            ))
            if user_vector.shape[1]<self.n_items:
                user_vector = sp.hstack((
                    user_vector,
                    sp.coo_matrix((1, self.n_items-user_vector.shape[1])),
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
            
    def build_item_features(self, item_features):
        feats_kostyl = sp.hstack((
            item_features,
            sp.coo_matrix((item_features.shape[0], item_features.shape[0]-item_features.shape[1])),
        ))
        return sp.identity(item_features.shape[0])+feats_kostyl
        
        
    def fit(self, ratings_data, algorithm, user_feats=None, item_feats=None):
        self.n_users = ratings_data['user_id'].nunique()
        self.n_items = 10000
        ratings_data_local = copy.deepcopy(ratings_data)
        ### reset user id
        if ratings_data_local.shape[0]<5976479:
            ratings_data_local['user_id'] = ratings_data_local['user_id'].replace(
                dict(zip(ratings_data_local['user_id'].unique(), np.arange(self.n_users))))
        ###
        self.algorithm = copy.deepcopy(algorithm)
        self.interactions, self.weights = self.build_interactions(ratings_data_local)
        if item_feats is not None: 
            self.item_features = self.build_item_features(item_feats)
        else: self.item_features = None
        
        interactions_upd, weights_upd = self.update_interacions()
        
        self.algorithm.fit(
            interactions = interactions_upd,
            #user_features = user_feats,
            item_features = self.item_features,
            sample_weight = weights_upd,
            epochs = 10
        )
        return self
        
    def predict_list(self, user_ratings, n=10, handle_series=False):
        interactions_new, weights_new = self.update_interacions(user_ratings)
        
        algorithm_local = copy.deepcopy(self.algorithm)
        algorithm_local.fit_partial(
            interactions = interactions_new,
            #user_features = user_features,
            item_features = self.item_features,
            sample_weight = weights_new
        )
        algorithm_local.item_biases = np.zeros_like(algorithm_local.item_biases)
        
        titles = pd.read_csv('data/books.csv')['title'].values
        known_positives = titles[weights_new.tocsr()[0].indices]
        scores = algorithm_local.predict(0, np.arange(self.n_items))
        top_items = titles[np.argsort(-scores)]
        
        return top_items[:n]

genres = sp.load_npz('data/genres.npz')
authors = sp.load_npz('data/authors.npz')
languages = sp.load_npz('data/languages.npz')
features = sp.hstack((
    #genres,
    authors,
    languages,
    #np.array(books['original_publication_year'])[:, None]
)).tocsr()

#model = LightFM(learning_rate=0.05, loss='bpr', random_state=1)
#lightfm = LightFM_Recommender()
#lightfm.fit(ratings_random, model)

#model = LightFM(learning_rate=0.05, loss='bpr', random_state=1)
#lightfm_hybrid = LightFM_Recommender()
#lightfm_hybrid.fit(ratings_random, model, item_feats = features)

with open('lightfm.pickle', 'rb') as f:
    lightfm = pickle.load(f)

with open('lightfm_hybrid.pickle', 'rb') as f:
    lightfm_hybrid = pickle.load(f)

def fetch_user_ratings_dataset(user_id):
    df = ratings.loc[ratings['user_id']==user_id, :]
    df['user_id'] = np.repeat(0, df.shape[0])
    return df

def fetch_user_ratings_goodreads(goodreads_id):
    import requests
    url = f"https://www.goodreads.com/review/list/{goodreads_id}?print=true"
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_10_1) AppleWebKit/537.36 \
    (KHTML, like Gecko) Chrome/39.0.2171.95 Safari/537.36'}
    html_output = requests.get(url=url, headers = headers).text

    import re
    num_books = int(re.findall('books on Goodreads \((.*) books\)', html_output)[0])

    for p in range(2, (num_books // 20 + (num_books % 20 > 0) + 1)):
        url = f"https://www.goodreads.com/review/list/{goodreads_id}?page={p}&print=true"
        html_output += requests.get(url=url, headers = headers).text

    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html_output, 'html.parser')

    bad_stuff_titles = ['Goodreads Home', None, 'My group discussions', 'Messages', 'Friends', 
                      'did not like it', 'it was ok', 'liked it', 'really liked it', 'it was amazing',]
    books = [i.get('title') for i in soup.find_all('a') 
             if i.get('title') not in bad_stuff_titles]
    
    stars = [i.get('class')[1] for i in soup.find_all('span') 
             if i.get('class') in [['staticStar', 'p0'], ['staticStar', 'p10']] ]
    
    def groupwise(iterable):
        a = iter(iterable)
        return zip(a, a, a, a, a)

    ratings = [(s1, s2, s3, s4, s5).count('p10') for (s1, s2, s3, s4, s5) in groupwise(stars)]
    
    book_map = pd.read_csv('data/books.csv')[['id', 'title', 'authors']]
    df = pd.merge(pd.DataFrame({'title' : books, 'rating' : ratings}), book_map, on='title')
    df = df.drop('title', axis=1).rename({'id' : 'book_id'}, axis=1)
    df['user_id'] = np.repeat(0, df.shape[0])
    df = df.reindex(columns=['user_id', 'book_id', 'rating'])
    df = df.loc[df['rating'] != 0]

    if df.empty:
        raise ValueError('No matching books rated')
    else: return df

def fancy_title(title):
    book_map = pd.read_csv('data/books.csv')[['title', 'authors']]
    return '<b>' + title + '</b>' + '\n' + str(book_map.loc[book_map['title']==title, 'authors'].values[0])# + '\n\n'

def fancy_list(reclist):
    text = ''
    for i, title in enumerate(reclist):
        text = text + fancy_title(title) + '\n\n'
        
    return text

from telegram import Update, ReplyKeyboardMarkup
from telegram.ext import (
    Updater,
    CommandHandler,
    MessageHandler,
    Filters,
    ConversationHandler,
    CallbackQueryHandler,
    CallbackContext,
)
from getpass import getpass

CHOOSING_SCENARIO, SELECTING_ENGINE = map(chr, range(2))
CHOOSING_USER_ID, CHOOSING_GR_ID, CHOOSING_CUSTOM = map(chr, range(2, 5))
TYPING_USER, TYPING_GOODREADS, TYPING_BOOK, SELECTING_RATING = map(chr, range(5, 9))

def start(update, context):
    context.user_data['user_ratings'] = None
    context.user_data['rated_dict'] = {}
    context.user_data['selected_book'] = None

    text = "Choose scenario"
    buttons = [[
                'Dataset Id',
                'GoodReads Id',
                'Custom Setup'
                ]]
    keyboard = ReplyKeyboardMarkup(buttons, one_time_keyboard=True)
    update.message.reply_text(text=text, reply_markup=keyboard)

    return CHOOSING_SCENARIO

def recommend(update, context):
    text = "Choose recommender engine"
    buttons = [['KNN', 'SVD', 'LightFM', 'Hybrid LightFM']]

    keyboard = ReplyKeyboardMarkup(buttons, one_time_keyboard=True)
    update.message.reply_text(text=text, reply_markup=keyboard)

    return SELECTING_ENGINE

def ask_user_id(update, context):
    text = 'Type a number between 0 and ' + str(ratings.user_id.nunique())
    update.message.reply_text(text=text)
    return TYPING_USER

def ask_user_id_again(update, context):
    text = 'Wrong input. Type a number between 0 and ' + str(ratings.user_id.nunique())
    update.message.reply_text(text=text)
    return TYPING_USER

def save_user_id(update, context):
    try:
        user_id = int(update.message.text)
        if user_id not in range(ratings.user_id.nunique()+1):
            raise ValueError
        context.user_data['user_ratings'] = fetch_user_ratings_dataset(user_id)
        return recommend(update, context)
    except ValueError:
        return ask_user_id_again(update, context)

def ask_goodreads_id(update, context):
    text = 'Type your GoodReads Id'
    update.message.reply_text(text=text)
    return TYPING_GOODREADS

def ask_goodreads_id_again(update, context):
    text = 'User does not exist or has no books from dataset rated. Please try again or choose different scenario by calling /start'
    update.message.reply_text(text=text)
    return TYPING_GOODREADS

def save_goodreads_id(update, context):
    try:
        user_id = int(update.message.text)
        context.user_data['user_ratings'] = fetch_user_ratings_goodreads(user_id)
        return recommend(update, context)
    except ValueError:
        return ask_goodreads_id_again(update, context)

def ask_book_rating(update, context):
    text = 'Type the book title you want to rate or finish rating'
    buttons = [['Finish']]
    keyboard = ReplyKeyboardMarkup(buttons, one_time_keyboard=True)
    update.message.reply_text(text=text, reply_markup=keyboard)
    return TYPING_BOOK

def save_selected_book(update, context):
    book_name = update.message.text
    book_map = pd.read_csv('data/books.csv')[['id', 'authors', 'title']]
    try:
        selected_book = process.extract(book_name, book_map['title'].values, scorer=fuzz.ratio)[0][0]
    except:
        update.message.reply_text(text='BAD')
    context.user_data['selected_book'] = selected_book
    
    return show_selected_book(update, context)

def show_selected_book(update, context):
    buttons = [['1','2','3','4','5'],
               ['Cancel', 'Finish']]
    keyboard = ReplyKeyboardMarkup(buttons, one_time_keyboard=True)
    update.message.reply_text(text=fancy_title(context.user_data['selected_book']), parse_mode = 'HTML')
    update.message.reply_text(text='Now rate it', reply_markup=keyboard)
    
    return SELECTING_RATING

def save_book_rating(update, context):
    book_rating = int(update.message.text)
    context.user_data['rated_dict'][context.user_data['selected_book']] = book_rating
    
    return ask_book_rating(update, context)

def rating_finished(update, context):
    books, ratings = zip(*context.user_data['rated_dict'].items())
    book_map = pd.read_csv('data/books.csv')[['id', 'title', 'authors']]
    book_ids = [book_map.loc[book_map['title']==i, 'id'].values[0] for i in books]
    context.user_data['user_ratings'] = pd.DataFrame(
        {'user_id' : [0]*len(book_ids), 'book_id' : book_ids, 'rating' : ratings})
    
    return recommend(update, context)

def rec_knn(update, context):
    warn = update.message.reply_text(text='This takes time')
    knn = KNNWithMeans(k=9, verbose=False)
    reclist = recommend_list(context.user_data['user_ratings'], ratings, knn, verbose = False)
    warn.edit_text(text=fancy_list(reclist), parse_mode = 'HTML')

def rec_svd(update, context):
    warn = update.message.reply_text(text='This takes time')
    svd = SVD(n_factors=20, verbose=False)
    reclist = recommend_list(context.user_data['user_ratings'], ratings, svd, verbose = False)
    warn.edit_text(text=fancy_list(reclist), parse_mode = 'HTML')

def rec_lightfm(update, context):
    warn = update.message.reply_text(text='This takes time')
    #reclist = recommend_list_lightfm(context.user_data['user_ratings'], ratings, model, verbose = False)
    reclist = lightfm.predict_list(context.user_data['user_ratings'])
    warn.edit_text(text=fancy_list(reclist), parse_mode = 'HTML')

def rec_lightfm_hybrid(update, context):
    warn = update.message.reply_text(text='This takes time')
    #reclist = recommend_list_lightfm(context.user_data['user_ratings'], ratings, model, verbose = False)
    reclist = lightfm_hybrid.predict_list(context.user_data['user_ratings'])
    warn.edit_text(text=fancy_list(reclist), parse_mode = 'HTML')

convhandler = ConversationHandler(
    entry_points = [CommandHandler('start', start)],
    states = {
        CHOOSING_SCENARIO: [MessageHandler(Filters.regex('^Dataset Id$'), ask_user_id),
                            MessageHandler(Filters.regex('^GoodReads Id$'), ask_goodreads_id),
                            MessageHandler(Filters.regex('^Custom Setup$'), ask_book_rating)],
        TYPING_USER: [MessageHandler(Filters.text & ~Filters.command, save_user_id)],
        TYPING_GOODREADS: [MessageHandler(Filters.text & ~Filters.command, save_goodreads_id)],
        TYPING_BOOK: [MessageHandler(Filters.text & ~Filters.command & ~Filters.regex('^Finish$'), save_selected_book),
                      MessageHandler(Filters.regex('^Finish$'), rating_finished)],
        SELECTING_RATING: [MessageHandler(Filters.regex('^[1-5]$'), save_book_rating),
                           MessageHandler(Filters.regex('^Finish$'), rating_finished),
                           MessageHandler(Filters.regex('^Cancel$'), ask_book_rating)],
        SELECTING_ENGINE: [MessageHandler(Filters.regex('^KNN$'), rec_knn),
                           MessageHandler(Filters.regex('^SVD$'), rec_svd),
                           MessageHandler(Filters.regex('^LightFM$'), rec_lightfm),
                           MessageHandler(Filters.regex('^Hybrid LightFM$'), rec_lightfm_hybrid)]
    },
    fallbacks = [CommandHandler('start', start)]
)

TOKEN = '1776136579:AAEkS7z3Lr3PrZMDMiXpWKD-OpR7P305K4M'
PORT = int(os.environ.get('PORT', '8443'))

def main():
    updater = Updater(token=TOKEN, use_context=True)
    dispatcher = updater.dispatcher
    dispatcher.add_handler(convhandler)
#    updater.start_polling()
    updater.start_webhook(listen="0.0.0.0",
                          port=PORT,
                          url_path=TOKEN,
			  webhook_url="https://goodbooks-bot.herokuapp.com/" + TOKEN)

    #updater.bot.set_webhook('goodbooks-bot' + '1776136579:AAEkS7z3Lr3PrZMDMiXpWKD-OpR7P305K4M')
    updater.idle()

if __name__ == '__main__':
    main()