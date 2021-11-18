import pandas as pd
import numpy as np
import scipy.sparse as sp
import time, copy, os, pickle, re
from lightfm import LightFM
from fuzzywuzzy import fuzz, process

from utils import (
    DatasetFaster,
    fetch_user_ratings_dataset,
    fetch_user_ratings_goodreads,
    predict_list,
    load_model,
    book_map,
    fancy_title,
    fancy_list
)

from config import *

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


CHOOSING_SCENARIO, SELECTING_ENGINE = map(chr, range(2))
CHOOSING_USER_ID, CHOOSING_GR_ID, CHOOSING_CUSTOM = map(chr, range(2, 5))
TYPING_USER, TYPING_GOODREADS, TYPING_BOOK, SELECTING_RATING = map(chr, range(5, 9))


def start(update, context):
    context.user_data['user_ratings'] = None
    context.user_data['rated_dict'] = {}
    context.user_data['selected_book'] = None
    
    update.message.reply_text(text=start_text, disable_web_page_preview=True)
    
    buttons = [[
        'Dataset Id',
        'GoodReads Id',
        'Custom Setup'
    ]]
    keyboard = ReplyKeyboardMarkup(buttons, one_time_keyboard=True)
    update.message.reply_text(text=scenario_text, reply_markup=keyboard)

    return CHOOSING_SCENARIO


def recommend(update, context):
    buttons = [[
        'LightFM',
        #'Hybrid LightFM'
    ]]
    keyboard = ReplyKeyboardMarkup(buttons, one_time_keyboard=True)
    update.message.reply_text(text=recommend_text, reply_markup=keyboard)

    return SELECTING_ENGINE


def ask_user_id(update, context):
    update.message.reply_text(text=id_text)
    
    return TYPING_USER


def ask_user_id_again(update, context):
    update.message.reply_text(text=wrongid_text)
    
    return TYPING_USER


def save_user_id(update, context):
    try:
        user_id = int(update.message.text)
        context.user_data['user_ratings'] = fetch_user_ratings_dataset(user_id)
        return recommend(update, context)
    except ValueError:
        return ask_user_id_again(update, context)


def ask_goodreads_id(update, context):
    update.message.reply_text(text=gr_text)
    
    return TYPING_GOODREADS


def ask_goodreads_id_again(update, context):
    update.message.reply_text(text=nouser_text)
    
    return TYPING_GOODREADS


def save_goodreads_id(update, context):
    try:
        goodreads_id = int(re.search('[0-9]+', update.message.text).group())
        context.user_data['user_ratings'] = fetch_user_ratings_goodreads(goodreads_id)
        return recommend(update, context)
    except:
        return ask_goodreads_id_again(update, context)

    
def inform_books_rating(update, context):
    update.message.reply_text(text=rating_text)
    
    return ask_book_rating(update, context)


def ask_book_rating(update, context):
    buttons = [['Finish']]
    keyboard = ReplyKeyboardMarkup(buttons, one_time_keyboard=True)
    update.message.reply_text(text=askbook_text, reply_markup=keyboard)
    
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
    update.message.reply_text(text=nowrate_text, reply_markup=keyboard)

    return SELECTING_RATING


def save_book_rating(update, context):
    book_rating = int(update.message.text)
    context.user_data['rated_dict'][context.user_data['selected_book']] = book_rating

    return ask_book_rating(update, context)


def rating_finished(update, context):
    books, rates = zip(*context.user_data['rated_dict'].items())
    book_ids = book_map['id'][book_map['title'].isin(books)].values

    context.user_data['user_ratings'] = pd.DataFrame(
        {'user_id': np.repeat(0, len(book_ids)), 'book_id': book_ids, 'rating': rates})

    return recommend(update, context)


def rec_lightfm(update, context):
    lightfm = load_model('lightfm1.pickle')
    warn = update.message.reply_text(text=wait_text)
    
    ratings = pd.read_csv('data/ratings.csv')
    dataset = DatasetFaster()
    dataset.fit(ratings.user_id.nunique(), 10000)
    interactions_new, weights_new = dataset.build_interactions(ratings, context.user_data['user_ratings'])

    lightfm.fit_partial(
        interactions=interactions_new,
        sample_weight=weights_new
    )
    
    reclist = predict_list(lightfm, context.user_data['user_ratings'])
    # warn.edit_text(text=fancy_list(reclist), parse_mode='HTML', disable_web_page_preview=True)
    rec = update.message.reply_text(text=fancy_list(reclist), parse_mode='HTML', disable_web_page_preview=True)


convhandler = ConversationHandler(
    entry_points=[CommandHandler('start', start)],
    states={
        CHOOSING_SCENARIO: [
            MessageHandler(Filters.regex('^Dataset Id$'), ask_user_id),
            MessageHandler(Filters.regex('^GoodReads Id$'), ask_goodreads_id),
            MessageHandler(Filters.regex('^Custom Setup$'), inform_books_rating)
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
            MessageHandler(Filters.regex('^LightFM$'), rec_lightfm),
            # MessageHandler(Filters.regex('^Hybrid LightFM$'), rec_lightfm_hybrid)
        ]
    },
    fallbacks=[CommandHandler('start', start)]
)


def main():
    TOKEN = os.environ['TOKEN']
    PORT = int(os.environ.get('PORT', '8443'))
    
    updater = Updater(token=TOKEN, use_context=True)
    dispatcher = updater.dispatcher
    dispatcher.add_handler(convhandler)
    updater.start_webhook(
        listen='0.0.0.0',
        port=PORT,
        url_path=TOKEN,
        webhook_url='https://goodbooks-bot.herokuapp.com/' + TOKEN
    )
    updater.idle()


if __name__ == '__main__':
    main()
