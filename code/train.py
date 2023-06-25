import sys
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback
)
import pandas as pd
from datasets import DatasetDict, Dataset
import evaluate
import numpy as np
from transformers.integrations import TensorBoardCallback
import os

def train(model_name, dataset_path, test_split=0.3, learning_rate=2e-5, epochs=10):

  generator_model_name = model_name
  df = pd.read_csv(dataset_path)
  test_split = test_split

  movies = Dataset.from_pandas(df)

  tokenizer = AutoTokenizer.from_pretrained(generator_model_name)

  def preprocess(examples):
      t5_task_prefix = "summarize: " 
      inputs = [t5_task_prefix + doc for doc in examples["overview"]]
      model_inputs = tokenizer(inputs, truncation=True)
      labels = tokenizer(examples["title"], truncation=True, max_length=10)

      model_inputs["labels"] = labels["input_ids"]
      return model_inputs

  tokenized_overviews = movies.map(preprocess, batched=True)
  train_test = tokenized_overviews.train_test_split(shuffle = True, test_size=test_split, seed=1)
  # test_valid = train_testvalid["test"].train_test_split(shuffle = True, test_size=(test_split/(test_split + val_split)), seed=random_state)

  model = AutoModelForSeq2SeqLM.from_pretrained(generator_model_name)
  data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=generator_model_name)

  tokenized_overviews = DatasetDict(
      train = train_test["train"],
      test = train_test["test"],
  )


  metrics = evaluate.load("rouge")

  def compute_metrics(eval_pred):
      predictions, labels = eval_pred
      decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True, batch_size=1)
      labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
      decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True, batch_size=1)

      result = metrics.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

      prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
      result["gen_len"] = np.mean(prediction_lens)
      return {k: round(v, 4) for k, v in result.items()}

  output_dir = f"/opt/movies/runs/{model_name}_lr{learning_rate}_epochs{epochs}"

  training_args = Seq2SeqTrainingArguments(
      output_dir=f"{output_dir}/train-results",
      evaluation_strategy="steps",
      learning_rate=learning_rate,
      per_device_train_batch_size=48,
      per_device_eval_batch_size=48,
      weight_decay=0.01,
      save_total_limit=3,
      logging_steps=50,
      logging_dir=f"{output_dir}/logs",
      logging_strategy="steps",
      num_train_epochs=epochs,
      predict_with_generate=True,
      fp16=True,
      push_to_hub=False,
      do_eval=True,
      eval_steps=50,
  )

  trainer = Seq2SeqTrainer(
      model=model,
      args=training_args,
      train_dataset=tokenized_overviews["train"],
      eval_dataset=tokenized_overviews["test"],
      tokenizer=tokenizer,
      data_collator=data_collator,
      compute_metrics=compute_metrics,
  )

  trainer.train()

  trainer.save_model(f"{output_dir}/model")

if __name__ == "__main__":
  model_name = "facebook/bart-base"#"t5-base"
  dataset_path = "/opt/movies/data/dataset_True_True_True_True_True.csv"
  test_split = 0.3
  learning_rate = 2e-5
  epochs = 2
  train(model_name, dataset_path, test_split, learning_rate, epochs)