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

generator_model_name = "t5-base"
df = pd.read_csv('/opt/movies/data/dataset_True_True_True_True_True.csv')

test_split = 0.3

movies = Dataset.from_pandas(df)

tokenizer = AutoTokenizer.from_pretrained(generator_model_name)

def preprocess(examples):
    t5_task_prefix = "summarize: " 
    inputs = [t5_task_prefix + doc for doc in examples["overview"]]
    model_inputs = tokenizer(inputs, truncation=True)
    labels = tokenizer(examples["title"], truncation=True, max_length=10)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_subtitles = movies.map(preprocess, batched=True)
train_test = tokenized_subtitles.train_test_split(shuffle = True, test_size=test_split, seed=1)
# test_valid = train_testvalid["test"].train_test_split(shuffle = True, test_size=(test_split/(test_split + val_split)), seed=random_state)

model = AutoModelForSeq2SeqLM.from_pretrained(generator_model_name)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=generator_model_name)

tokenized_subtitles = DatasetDict(
    train = train_test["train"],
    test = train_test["test"],
)


metrics = evaluate.load("rouge")

def compute_metrics(eval_pred):
    print("COMPUTE METRICS")
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True, batch_size=1)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True, batch_size=1)

    result = metrics.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)
    print("COMPUTE METRICS END")
    return {k: round(v, 4) for k, v in result.items()}


training_args = Seq2SeqTrainingArguments(
    output_dir="movie-overview-predictor",
    evaluation_strategy="epoch",
    learning_rate=1e-6,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3000,
    predict_with_generate=True,
    fp16=True,
    push_to_hub=False
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_subtitles["train"],
    eval_dataset=tokenized_subtitles["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()

trainer.save_model("/opt/movies/model-backup/movie-overview-predictor")

print("LO DEJE CORRIENDO A LAS 3:10")

# subtitle = df_total['overview'][20]
# print(len(subtitle))
# text = "summarize: " + subtitle
# inputs = tokenizer.encode(text, return_tensors="pt", truncation=False).to("cuda")
# print(inputs)
# outputs = model.generate(inputs, do_sample=False, num_beams=3)
# output = tokenizer.decode(outputs[0], skip_special_tokens=True)
# print("EXPECTED " + subtitle)
# print("PREDICTED " + output)