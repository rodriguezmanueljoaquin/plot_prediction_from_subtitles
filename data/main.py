from overview_preprocess import process_dataset
from analysis import analysis

if __name__ == '__main__':
    data_df = process_dataset(to_lower_case=True, remove_contractions=True, remove_stopwords=True,\
             remove_symbols=True, apply_lemmatization=True)
    
