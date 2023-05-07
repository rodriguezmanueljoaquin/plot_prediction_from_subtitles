from dataset.generate_dataset import generate_dataset
from analysis import analysis

if __name__ == '__main__':
    data_df = generate_dataset(to_lower_case=False, with_contractions=False, with_stopwords=False,\
             with_symbols=False, with_lemmatization=False)

    analysis(data_df)


