import sys
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
import pandas as pd
from datasets import DatasetDict, Dataset
import evaluate
import numpy as np
import os

df = pd.read_csv('/opt/movies/data/dataset_True_True_True_True_True.csv')

movies = Dataset.from_pandas(df)
train_test = movies.train_test_split(shuffle = True, test_size=0.2, seed=1)
df = train_test["test"].to_pandas()

def test_model(run_path, metric='rouge'):
  generator_run_path = run_path# "t5-base"#"../runs/t5-base/model"
  tokenizer = AutoTokenizer.from_pretrained(f'{generator_run_path}/model')
  model = AutoModelForSeq2SeqLM.from_pretrained(f'{generator_run_path}/model', return_dict=True).to('cuda')

  def evaluate_fun():
    predictions = []
    targets = []
    for serie in df.itertuples():
      text = f'summarize: {serie.overview}'
      inputs = tokenizer.encode(text, return_tensors="pt", truncation=False).to('cuda')
      outputs = model.generate(inputs, do_sample=False, num_beams=3, max_length=10, early_stopping=True)
      output = tokenizer.decode(outputs[0], skip_special_tokens=True)
      # print(f'expected: {serie.title} - predicted: {output}')
      predictions.append(output)
      targets.append(serie.title)
    
    return predictions, targets

  predictions, targets = evaluate_fun()

  # save csv with predictions and targets
  test_df = pd.DataFrame({'predictions': predictions, 'targets': targets})
  test_df.to_csv(f'{run_path}/test.csv', index=False)

  metrics = evaluate.load(metric)
  results = metrics.compute(predictions=predictions, references=targets, use_stemmer=True)
  # save results
  with open(f'{run_path}/results.txt', 'w') as f:
    f.write(str(results))
  
if __name__ == '__main__':
  # iterate through all files in runs folder
  for run_path in os.listdir('/opt/movies/runs'):
    if 'google' in run_path:
      continue
    test_model(f'/opt/movies/runs/{run_path}')