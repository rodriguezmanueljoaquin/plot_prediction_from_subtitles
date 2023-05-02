import pandas as pd
import re

def generate_dataset():
    meta_df = pd.read_csv('./data/movies_metadata.csv')
    subtitles_with_time_df = pd.read_csv('./data/movies_subtitles.csv')

    # drop duplicates in meta_df
    meta_df.drop_duplicates(subset='imdb_id', keep="first", inplace=True)

    # drop rows with original language different from english
    meta_df = meta_df[meta_df['original_language'] == 'en']

    meta_df.drop(columns=['adult', 'belongs_to_collection', 'budget', 'genres', 'homepage', 'id', 'original_language'], inplace=True)
    
    # drop registries with missing values
    meta_df.dropna(inplace=True)
    subtitles_with_time_df.dropna(inplace=True)

    print("Total of {} subtitles before processing".format(subtitles_with_time_df.size))

    # remove subtitles with url
    subtitles_df = process_subtitles_that_match(subtitles_with_time_df, re.compile(r'www.'), replace=False)

    # remove html with regex
    subtitles_df = process_subtitles_that_match(subtitles_df, re.compile(r'<[^>]*>'), replace=True)

    # remove non closed captions symbols
    subtitles_df = process_subtitles_that_match(subtitles_df, re.compile(r'[^a-zA-Z0-9!@#$%^&*()_+={\[}\]|:;"\',.?/\- ]'), replace=True)

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

    print("Removing {} subtitles with less than 10% of the mean words in subtitles".format(merged_df[merged_df['subtitles_word_count'] < mean_words].size))
    # discard those with less than mean words
    merged_df = merged_df[merged_df['subtitles_word_count'] >= mean_words]


    print("--------------------")
    print("Total of {} movies".format(merged_df.size))
    print(merged_df.tail())


def process_subtitles_that_match(df, regex_pattern, replace=True):
    subtitles_to_remove = df['text'].str.contains(regex_pattern, regex=True)
    if replace:
        df.loc[subtitles_to_remove, 'text'] = df['text'].str.replace(regex_pattern, '', regex=True)
    else:
        df = df[~subtitles_to_remove]

    print(f'Re{"plac" if replace else "mov"}ing {subtitles_to_remove.count()} subtitles with regex pattern {regex_pattern}')
    return df
