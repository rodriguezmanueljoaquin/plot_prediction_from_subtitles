import sys
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
)
from datasets import DatasetDict, Dataset
import numpy as np
import pandas as pd
import evaluate
import os
from bert_score import score

df = pd.read_csv('/opt/movies/data/dataset_True_True_True_True_True.csv')

movies = Dataset.from_pandas(df)
train_test = movies.train_test_split(shuffle = True, test_size=0.2, seed=1)
df = train_test["test"].to_pandas()
"""
def test_model(run_path, model_name, metric="bleu"):
  # run_path = "/opt/movies/runs/google_pegasus-large_lr2e-05_epochs50"
  generator_run_path = run_path# "t5-base"#"../runs/t5-base/model"
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForSeq2SeqLM.from_pretrained(model_name, return_dict=True).to('cuda')

  def evaluate_fun():
    predictions = []
    targets = []
    for serie in df.itertuples():
      text = f'summarize: {serie.overview}' if model_name == 't5-base' else serie.overview
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
  test_df.to_csv(f'{run_path}/{metric}-test.csv', index=False)

  metrics = evaluate.load(metric)
  results = metrics.compute(predictions=predictions, references=targets)
  # save results
  with open(f'{run_path}/{metric}-results.txt', 'w') as f:
    f.write(str(results))
  
"""
def evaluate_metric(run_path, metric):
  # load csv with predictions and targets
  test_df = pd.read_csv(f'{run_path}/test.csv')
  predictions = test_df['predictions'].tolist()
  targets = test_df['targets'].tolist()

  if metric == 'bertscore':
    P, R, F1 = score(predictions, targets, lang="en", verbose=True)
    results = {'precision': P.mean().item(), 'recall': R.mean().item(), 'f1': F1.mean().item()}
  else:
    metrics = evaluate.load(metric)
    results = metrics.compute(predictions=predictions, references=targets)
  
  # save results
  with open(f'{run_path}/{metric}-results.txt', 'w') as f:
    f.write(str(results))

if __name__ == '__main__':
  metrics = ["rouge", "bleu", "chrf", "bertscore"]
  # iterate through all files in runs folder
  for run_path in os.listdir('/opt/movies/runs/base'):
    # test_model(f'/opt/movies/runs/base/{run_path}', run_path.replace('_', '/'))
    # if not run_path.endswith('base'):
      # continue
    evaluate_metric(f'/opt/movies/runs/base/{run_path}', metrics[3])
    # evaluate_metric(f'/opt/movies/runs/{run_path}', metrics[3])
