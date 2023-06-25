import pandas as pd
import re
import contractions
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')

column_to_process = 'overview'

def process_dataset(to_lower_case=False, remove_contractions=False, remove_stopwords=False,\
             remove_symbols=False, apply_lemmatization=False):
    meta_df = pd.read_csv('./original_dataset/movies_metadata.csv')

    print("Total of {} movies before preprocessing".format(len(meta_df)))

    # drop duplicates in meta_df
    print("Total of {} duplicates in meta_df".format(meta_df.duplicated().sum()))
    meta_df.drop_duplicates(subset='imdb_id', keep="first", inplace=True)

    meta_df.drop(
        columns=['adult', 'belongs_to_collection', 'budget',
                 'genres', 'homepage', 'id', 'original_language', 
                 'popularity','poster_path','production_companies',
                 'production_countries','release_date','revenue',
                 'runtime','spoken_languages','status','tagline',
                 'video','vote_average','vote_count'],
        inplace=True)

    # drop registries with missing values
    print("Total of {} registries with missing values in movies metadata".format(
        meta_df.isnull().sum().sum()))
    meta_df.dropna(inplace=True)

    # remove overview with url
    meta_df = process_overviews_that_match(
        meta_df, re.compile(r'www\.'), replace=False)

    # remove html with regex
    meta_df = process_overviews_that_match(
        meta_df, re.compile(r'<[^>]*>'), replace=True)

    # replace \n with space in text
    meta_df = process_overviews_that_match(
        meta_df, re.compile(r'\n'), replace=True, replace_with=' ')

    # remove non closed captions symbols
    meta_df = process_overviews_that_match(meta_df,
                                                re.compile(r'[^a-zA-Z0-9!@#$%^&*()_+={\[}\]|:;"\',.?/\-\n ]'), replace=True)

    # remove empty overviews
    meta_df = process_overviews_that_match(meta_df,
                                    re.compile(r'No overview found.'), replace=False)

    print("--------------------")
    print("Total of {} movies".format(len(meta_df)))
    print(meta_df.tail())

    if to_lower_case:
        meta_df.loc[:, 'overview'] = meta_df['overview'].str.lower()
        print("Overview transformed to lower case")

    if remove_symbols:
        meta_df.loc[:, 'overview'] = meta_df['overview']\
            .apply(lambda x: re.sub(r'[@#$%^&*()_+={\[}\]|:;"\',.?/\-]', ' ', x))
        print("Overview symbols removed")

    if remove_contractions:
        meta_df.loc[:, 'overview'] = meta_df['overview']\
            .apply(lambda x: contractions.fix(x))
        print("Overview contractions expanded")

    if remove_stopwords:
        eng_stopwords = stopwords.words('english')
        meta_df.loc[:, 'overview'] = meta_df['overview']\
            .apply(lambda x: ' '.join([word for word in x.split() if word not in eng_stopwords]))
        print("Overview stopwords removed")

    if apply_lemmatization:
        meta_df.loc[:, 'overview'] = meta_df['overview']\
            .apply(lambda x: ' '.join([WordNetLemmatizer().lemmatize(word) for word in word_tokenize(x)]))
        print("Overview lemmatized")


    print("Overview transformed with to_lower_case={}, remove_contractions={}, remove_stopwords={}, remove_symbols={}, apply_lemmatization={}"\
            .format(to_lower_case, remove_contractions, remove_stopwords, remove_symbols, apply_lemmatization))
    print(meta_df.tail())

    meta_df['overview_word_count'] = meta_df['overview'].apply(
        lambda x: len(x.split()))
    min_word_count = 22
    print("Removing {} movies with less than {} words in overview"
          .format(len(meta_df[meta_df['overview_word_count'] <= min_word_count]), min_word_count))
    meta_df = meta_df[meta_df['overview_word_count'] > min_word_count]

    # save meta_df to csv
    name = 'dataset_{}_{}_{}_{}_{}.csv'.format(to_lower_case, remove_contractions, remove_stopwords, remove_symbols, apply_lemmatization)

    # check if file exists
    try:
        pd.read_csv('./' + name)
        print("File {} already exists".format(name))
    except:
        meta_df.to_csv('./' + name, index=False)
        print("File {} created".format(name))

    return meta_df


def process_overviews_that_match(df, regex_pattern, replace=True, replace_with=''):
    overviews_to_update = df['overview'].str.contains(regex_pattern, regex=True)
    if replace:
        df.loc[overviews_to_update, 'overview'] = df['overview'].str.replace(
            regex_pattern, replace_with, regex=True)
    else:
        df = df[~overviews_to_update]

    print(f'Re{"plac" if replace else "mov"}ing {overviews_to_update.sum()} overview with regex pattern {regex_pattern}')
    return df
