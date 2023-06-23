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

def generate_dataset(to_lower_case=False, remove_contractions=False, remove_stopwords=False,\
             remove_symbols=False, apply_lemmatization=False):
    meta_df = pd.read_csv('./dataset/movies_metadata.csv')
    subtitles_with_time_df = pd.read_csv(
        './dataset/movies_subtitles.csv', quotechar='"')

    # quantity of subtitles
    print("Total of {} subtitles".format(len(subtitles_with_time_df)))

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
    print("Total of {} registries with missing values in movies subtitles".format(
        subtitles_with_time_df.isnull().sum().sum()))
    subtitles_with_time_df.dropna(inplace=True)

    print("Total of {} subtitles before processing".format(
        len(subtitles_with_time_df)))

    # remove subtitles with url
    subtitles_df = process_subtitles_that_match(
        subtitles_with_time_df, re.compile(r'www\.'), replace=False)

    # remove html with regex
    subtitles_df = process_subtitles_that_match(
        subtitles_df, re.compile(r'<[^>]*>'), replace=True)

    # replace \n with space in text of subtitles
    subtitles_df = process_subtitles_that_match(
        subtitles_df, re.compile(r'\n'), replace=True, replace_with=' ')

    # remove non closed captions symbols
    subtitles_df = process_subtitles_that_match(subtitles_df,
                                                re.compile(r'[^a-zA-Z0-9!@#$%^&*()_+={\[}\]|:;"\',.?/\-\n ]'), replace=True)

    # order by imdb_id, then start time
    subtitles_with_time_sorted_df = subtitles_df.sort_values(
        by=['imdb_id', 'start_time'])

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
    merged_df['subtitles_word_count'] = merged_df['subtitles'].apply(
        lambda x: len(x.split()))

    # print("Removing {} movies with less than 30% of the mean words in subtitles"
    #       .format(len(merged_df[merged_df['subtitles_word_count'] < mean_words * 0.3])))
    # # discard those with less than mean * 0.3 words
    # final_df = merged_df[merged_df['subtitles_word_count'] >= mean_words * 0.3]

    print("Removing {} movies with more than 16000 words in subtitles"
          .format(len(merged_df[merged_df['subtitles_word_count'] > 16000])))
    # discard those with more than mean * 2 words
    final_df = merged_df[merged_df['subtitles_word_count'] <= 16000]

    print("--------------------")
    print("Total of {} movies".format(len(final_df)))
    print(final_df.tail())

    if to_lower_case:
        final_df.loc[:, 'subtitles'] = final_df['subtitles'].str.lower()
        print("Subtitles transformed to lower case")

    if remove_symbols:
        final_df.loc[:, 'subtitles'] = final_df['subtitles']\
            .apply(lambda x: re.sub(r'[@#$%^&*()_+={\[}\]|:;"\',.?/\-]', ' ', x))
        print("Subtitles symbols removed")

    if remove_contractions:
        final_df.loc[:, 'subtitles'] = final_df['subtitles']\
            .apply(lambda x: contractions.fix(x))
        print("Subtitles contractions expanded")

    if remove_stopwords:
        eng_stopwords = stopwords.words('english')
        final_df.loc[:, 'subtitles'] = final_df['subtitles']\
            .apply(lambda x: ' '.join([word for word in x.split() if word not in eng_stopwords]))
        print("Subtitles stopwords removed")

    if apply_lemmatization:
        final_df.loc[:, 'subtitles'] = final_df['subtitles']\
            .apply(lambda x: ' '.join([WordNetLemmatizer().lemmatize(word) for word in word_tokenize(x)]))
        print("Subtitles lemmatized")


    print("Subtitles transformed with to_lower_case={}, remove_contractions={}, remove_stopwords={}, remove_symbols={}, apply_lemmatization={}"\
            .format(to_lower_case, remove_contractions, remove_stopwords, remove_symbols, apply_lemmatization))
    print(final_df.tail())

    # update subtitles_word_count
    final_df['subtitles_word_count'] = final_df['subtitles'].apply(
        lambda x: len(x.split()))

    # save final_df to csv
    name = 'dataset_{}_{}_{}_{}_{}.csv'.format(to_lower_case, remove_contractions, remove_stopwords, remove_symbols, apply_lemmatization)

    # check if file exists
    try:
        pd.read_csv('./dataset/' + name)
        print("File {} already exists".format(name))
    except:
        final_df.to_csv('./dataset/' + name, index=False)
        print("File {} created".format(name))

    return final_df


def process_subtitles_that_match(df, regex_pattern, replace=True, replace_with=''):
    subtitles_to_update = df['text'].str.contains(regex_pattern, regex=True)
    if replace:
        df.loc[subtitles_to_update, 'text'] = df['text'].str.replace(
            regex_pattern, replace_with, regex=True)
    else:
        df = df[~subtitles_to_update]

    print(f'Re{"plac" if replace else "mov"}ing {subtitles_to_update.sum()} subtitles with regex pattern {regex_pattern}')
    return df
