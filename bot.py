import pandas as pd
import numpy as np
import scipy.sparse as sp
import time
import copy
import os
import pickle
#from surprise import Reader
#from surprise import Dataset as SurpriseDataset
#from surprise import KNNWithMeans, SVD, SVDpp
from lightfm import LightFM
from fuzzywuzzy import fuzz, process

from thisproject import (
    LightFM_Recommender,
    recommend_list,
    fetch_user_ratings_dataset,
    fetch_user_ratings_goodreads
)

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

pd.options.mode.chained_assignment = None

#ratings = pd.read_csv('data/ratings.csv')
book_map = pd.read_csv('data/books.csv')[['book_id', 'id', 'title', 'authors']]

#with open('lightfm.pickle', 'rb') as f:
#    lightfm = pickle.load(f)

#with open('lightfm_hybrid.pickle', 'rb') as f:
#    lightfm_hybrid = pickle.load(f)

    
def load_model(model_name):
    with open(str(model_name), 'rb') as f:
        model = pickle.load(f)
    return model


def fancy_title(title):
    # book_map = pd.read_csv('data/books.csv')[['id', 'book_id', 'title', 'authors']]
    site_id, author = book_map.loc[book_map['title'] == title, ['book_id', 'authors']].values[0]
    return '<b>' + f'<a href="https://www.goodreads.com/book/show/{site_id}">' + title + '</a>' + '</b>' + '\n' + author + '\n\n'


def fancy_list(reclist):
    return ''.join(map(fancy_title, reclist))


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
    buttons = [[
        # 'KNN',
        # 'SVD',
        'LightFM',
        'Hybrid LightFM'
    ]]
    keyboard = ReplyKeyboardMarkup(buttons, one_time_keyboard=True)
    update.message.reply_text(text=text, reply_markup=keyboard)

    return SELECTING_ENGINE


def ask_user_id(update, context):
    text = 'Type a number between 1 and ' + str(53424)
    update.message.reply_text(text=text)
    
    return TYPING_USER


def ask_user_id_again(update, context):
    text = 'Wrong input. Type a number between 1 and ' + str(53424)
    update.message.reply_text(text=text)
    
    return TYPING_USER


def save_user_id(update, context):
    try:
        user_id = int(update.message.text)
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
        goodreads_id = int(update.message.text)
        context.user_data['user_ratings'] = fetch_user_ratings_goodreads(goodreads_id)
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
    selected_book = process.extract(book_name, book_map['title'].values, scorer=fuzz.ratio)[0][0]
    context.user_data['selected_book'] = selected_book

    return show_selected_book(update, context)


def show_selected_book(update, context):
    buttons = [['1', '2', '3', '4', '5'],
               ['Cancel', 'Finish']]
    keyboard = ReplyKeyboardMarkup(buttons, one_time_keyboard=True)
    update.message.reply_text(text=fancy_title(context.user_data['selected_book']),
                              parse_mode='HTML',
                              disable_web_page_preview=True)
    update.message.reply_text(text='Now rate it', reply_markup=keyboard)

    return SELECTING_RATING


def save_book_rating(update, context):
    book_rating = int(update.message.text)
    context.user_data['rated_dict'][context.user_data['selected_book']] = book_rating

    return ask_book_rating(update, context)


def rating_finished(update, context):
    books, rates = zip(*context.user_data['rated_dict'].items())
    # book_map = pd.read_csv('data/books.csv')[['id', 'title', 'authors']]
    # book_ids = [book_map.loc[book_map['title'] == i, 'id'].values[0] for i in books]
    book_ids = book_map['id'][book_map['title'].isin(books)].values
    # book_ids = book_map.id.values[np.isin(book_map.title, books)]
    context.user_data['user_ratings'] = pd.DataFrame(
        {'user_id': np.repeat(0, len(book_ids)), 'book_id': book_ids, 'rating': rates})

    return recommend(update, context)


#def rec_knn(update, context):
#    warn = update.message.reply_text(text='This takes time')
#    knn = KNNWithMeans(k=9, verbose=False)
#    reclist = recommend_list(context.user_data['user_ratings'], ratings, knn, verbose=False)
#    warn.edit_text(text=fancy_list(reclist), parse_mode='HTML', disable_web_page_preview=True)


#def rec_svd(update, context):
#    warn = update.message.reply_text(text='This takes time')
#    svd = SVD(n_factors=20, verbose=False)
#    reclist = recommend_list(context.user_data['user_ratings'], ratings, svd, verbose=False)
#    warn.edit_text(text=fancy_list(reclist), parse_mode='HTML', disable_web_page_preview=True)


def rec_lightfm(update, context):
    lightfm = load_model('lightfm.pickle')
    warn = update.message.reply_text(text='This takes time')
    reclist = lightfm.predict_list(context.user_data['user_ratings'])
    warn.edit_text(text=fancy_list(reclist), parse_mode='HTML', disable_web_page_preview=True)


def rec_lightfm_hybrid(update, context):
    lightfm_hybrid = load_model('lightfm_hybrid.pickle')
    warn = update.message.reply_text(text='This takes time')
    reclist = lightfm_hybrid.predict_list(context.user_data['user_ratings'])
    warn.edit_text(text=fancy_list(reclist), parse_mode='HTML', disable_web_page_preview=True)


convhandler = ConversationHandler(
    entry_points=[CommandHandler('start', start)],
    states={
        CHOOSING_SCENARIO: [
            MessageHandler(Filters.regex('^Dataset Id$'), ask_user_id),
            MessageHandler(Filters.regex('^GoodReads Id$'), ask_goodreads_id),
            MessageHandler(Filters.regex('^Custom Setup$'), ask_book_rating)
        ],
        TYPING_USER: [
            MessageHandler(Filters.text & ~Filters.command, save_user_id)
        ],
        TYPING_GOODREADS: [
            MessageHandler(Filters.text & ~Filters.command, save_goodreads_id)
        ],
        TYPING_BOOK: [
            MessageHandler(Filters.text & ~Filters.command & ~Filters.regex('^Finish$'), save_selected_book),
            MessageHandler(Filters.regex('^Finish$'), rating_finished)
        ],
        SELECTING_RATING: [
            MessageHandler(Filters.regex('^[1-5]$'), save_book_rating),
            MessageHandler(Filters.regex('^Finish$'), rating_finished),
            MessageHandler(Filters.regex('^Cancel$'), ask_book_rating)
        ],
        SELECTING_ENGINE: [
            # MessageHandler(Filters.regex('^KNN$'), rec_knn),
            # MessageHandler(Filters.regex('^SVD$'), rec_svd),
            MessageHandler(Filters.regex('^LightFM$'), rec_lightfm),
            MessageHandler(Filters.regex('^Hybrid LightFM$'), rec_lightfm_hybrid)
        ]
    },
    fallbacks=[CommandHandler('start', start)]
)

TOKEN = os.environ['TOKEN']
PORT = int(os.environ.get('PORT', '8443'))


def main():
    updater = Updater(token=TOKEN, use_context=True)
    dispatcher = updater.dispatcher
    dispatcher.add_handler(convhandler)
    updater.start_webhook(listen='0.0.0.0',
                          port=PORT,
                          url_path=TOKEN,
                          webhook_url='https://goodbooks-bot.herokuapp.com/' + TOKEN)
    updater.idle()


if __name__ == '__main__':
    main()
