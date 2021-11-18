start_text = '''
This is an interactive recommendation system for the books on goodreads. See github.com/yuasosnin/minor-goodbooks-recommender
'''

recommend_text = 'Choose a recommender algorithm. Currently only one is available'

rating_text = '''
Type in the title of the book you want to rate. The bot will response with the closest find in the dataset and ask a rating. After that you can continue rating or finish and proceed to recommenation. It is recommended to rate several books.
'''

scenario_text = '''You can choose one of available scenarios:
    Dataset ID – recommend books for one of users from dataset (for demonstration purposes)
    GoodReads ID – your Goodreads profile ID or link
    Custom Setup – interactive rating process
'''
nouser_text = 'User does not exist or has no books from dataset rated. Please try again or choose a different scenario by calling /start'
gr_text = 'Paste a link to your Goodreads profile'
askbook_text = 'Type next book name or finish rating'
nowrate_text = 'Now rate the book'
wait_text = 'This takes some time'
id_text = 'Type in a number between 1 and ' + str(53424)
wrongid_text = 'Wrong number. Please type in a number between 1 and ' + str(53424)