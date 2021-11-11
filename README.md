# A recommendation system for Goodreads books

## Description

This is the final project for minor "Introduction to Data Analysis" at HSE. The task was to create an interactive application of something Data Science related.

The the app is a Telegram bot, which performs a book recommendation. The output is based on either an interactive rating process inside the app or a [goodreads](https://www.goodreads.com/) profile. Used algorithm is a [Light.FM](https://github.com/lyst/lightfm) matrix factorization algorithm, trained on 10000 books rated on goodreads ([data source](https://www.kaggle.com/zygmunt/goodbooks-10k)).

The bot itself is hosted on heroku.

## Usage

Find @BookRecBot in Telegram and follow instructions to receive a recommendation. (The process may take a while due to quite heavy computations.)
