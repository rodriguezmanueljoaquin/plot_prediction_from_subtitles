from dataset.generate_dataset import generate_dataset

if __name__ == '__main__':
    generate_dataset(to_lower_case=True, with_contractions=True, with_stopwords=True,\
             with_symbols=True, with_lemmatization=True)
