from dataset.generate_dataset import generate_dataset
from analysis import analysis

if __name__ == '__main__':
    data_df = generate_dataset(to_lower_case=True, remove_contractions=True, remove_stopwords=True,\
             remove_symbols=True, apply_lemmatization=True)
    
    data_df.to_csv('./dataset/dataset_True_True_True_True_True.csv', index=False)

    analysis(data_df)
