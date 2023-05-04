import pandas as pd
import re
import contractions
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
import nltk
nltk.download('stopwords')
nltk.download('wordnet')

def generate_dataset(to_lower_case=False, with_contractions=False, with_stopwords=False,\
             with_symbols=False, with_lemmatization=False):
    meta_df = pd.read_csv('./dataset/movies_metadata.csv')
    subtitles_with_time_df = pd.read_csv('./dataset/movies_subtitles.csv')

    # drop duplicates in meta_df
    meta_df.drop_duplicates(subset='imdb_id', keep="first", inplace=True)

    # drop rows with original language different from english
    meta_df = meta_df[meta_df['original_language'] == 'en']

    meta_df.drop(\
            columns=['adult', 'belongs_to_collection', 'budget', 'genres', 'homepage', 'id', 'original_language'],\
             inplace=True)
    
    # drop registries with missing values
    meta_df.dropna(inplace=True)
    subtitles_with_time_df.dropna(inplace=True)

    print("Total of {} subtitles before processing".format(subtitles_with_time_df.size))

    # remove subtitles with url
    subtitles_df = process_subtitles_that_match(subtitles_with_time_df, re.compile(r'www.'), replace=False)

    # remove html with regex
    subtitles_df = process_subtitles_that_match(subtitles_df, re.compile(r'<[^>]*>'), replace=True)

    # remove non closed captions symbols
    subtitles_df = process_subtitles_that_match(subtitles_df,\
             re.compile(r'[^a-zA-Z0-9!@#$%^&*()_+={\[}\]|:;"\',.?/\- ]'), replace=True)

    # order by imdb_id, then start time
    subtitles_with_time_sorted_df = subtitles_df.sort_values(by=['imdb_id', 'start_time'])

    # from subtitles_with_time_sorted_df group texts by imdb_id
    subtitles_grouped_df = subtitles_with_time_sorted_df \
            .groupby('imdb_id')['text'].apply(lambda x: '\n'.join(x)).reset_index()
    
    # change column name text to subtitles
    subtitles_grouped_df.rename(columns={'text': 'subtitles'}, inplace=True)
    
    # merge meta_df with subtitles_grouped_df
    merged_df = pd.merge(meta_df, subtitles_grouped_df, on='imdb_id')

    # discard those with less than 10% of the mean words in subtitles
    mean_words = merged_df['subtitles'].apply(lambda x: len(x.split())).mean()

    # create subtitles_word_count
    merged_df['subtitles_word_count'] = merged_df['subtitles'].apply(lambda x: len(x.split()))

    print("Removing {} subtitles with less than 10% of the mean words in subtitles"\
            .format(merged_df[merged_df['subtitles_word_count'] < mean_words].size))
    # discard those with less than mean words
    final_df = merged_df[merged_df['subtitles_word_count'] >= mean_words]


    print("--------------------")
    print("Total of {} movies".format(final_df.size))
    print(final_df.tail())

    if to_lower_case:
        final_df.loc[:, 'subtitles'] = final_df['subtitles'].apply(lambda x: x.lower())
        print("Subtitles transformed to lower case")

    if with_symbols:
        final_df.loc[:, 'subtitles'] = final_df['subtitles']\
            .apply(lambda x: re.sub(r'[@#$%^&*()_+={\[}\]|:;"\',.?/\-]', ' ', x))
        print("Subtitles symbols removed")

    if with_contractions:
        final_df.loc[:, 'subtitles'] = final_df['subtitles']\
            .apply(lambda x: contractions.fix(x))
        print("Subtitles contractions expanded")

    if with_stopwords:
        final_df.loc[:, 'subtitles'] = final_df['subtitles']\
            .apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords.words('english')]))
        print("Subtitles stopwords removed")

    if with_lemmatization:
        final_df.loc[:, 'subtitles'] = final_df['subtitles']\
            .apply(lambda x: ' '.join([WordNetLemmatizer().lemmatize(word) for word in  word_tokenize(x)]))
        print("Subtitles lemmatized")


    print("Subtitles transformed with to_lower_case={}, with_contractions={}, with_stopwords={}, with_symbols={}, with_lemmatization={}"\
            .format(to_lower_case, with_contractions, with_stopwords, with_symbols, with_lemmatization))
    print(final_df.tail())


def process_subtitles_that_match(df, regex_pattern, replace=True):
    subtitles_to_update = df['text'].str.contains(regex_pattern, regex=True)
    if replace:
        df.loc[subtitles_to_update, 'text'] = df['text'].str.replace(regex_pattern, '', regex=True)
    else:
        df = df[~subtitles_to_update]

    print(f'Re{"plac" if replace else "mov"}ing {subtitles_to_update.count()} \
            subtitles with regex pattern {regex_pattern}')
    return df
