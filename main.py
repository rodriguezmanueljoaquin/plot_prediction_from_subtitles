from dataset.generate_dataset import generate_dataset
from analysis import analysis

if __name__ == '__main__':
    data_df = generate_dataset(to_lower_case=False, remove_contractions=False, remove_stopwords=False,\
             remove_symbols=False, apply_lemmatization=False)

    analysis(data_df)
