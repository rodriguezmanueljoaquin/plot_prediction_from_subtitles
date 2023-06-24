from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    LongT5Model,
    Data
)
import pandas as pd
from datasets import DatasetDict, Dataset
import evaluate
import numpy as np

generator_model_name = "google/long-t5-tglobal-large"
df = pd.read_csv('../data/dataset_True_True_True_True_True.csv')
test_split = 0.3

movies = Dataset.from_pandas(df)

tokenizer = AutoTokenizer.from_pretrained(generator_model_name)

def preprocess(examples):
    t5_task_prefix = "summarize: " 
    inputs = [t5_task_prefix + doc for doc in examples["subtitles"]]
    model_inputs = tokenizer(inputs)

    labels = tokenizer(text_target=examples["overview"])

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_subtitles = movies.map(preprocess, batched=True, num_proc=4, remove_columns=[])

train_test = tokenized_subtitles.train_test_split(shuffle = True, test_size=test_split, seed=1)
# test_valid = train_testvalid["test"].train_test_split(shuffle = True, test_size=(test_split/(test_split + val_split)), seed=random_state)

tokenized_subtitles = DatasetDict(
    train = train_test["train"],
    test = train_test["test"],
)

metrics = evaluate.load("rouge")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    result = metrics.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
    result["gen_len"] = np.mean(prediction_lens)

    return {k: round(v, 4) for k, v in result.items()}

model = LongT5Model.from_pretrained(generator_model_name)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir="movie-overview-predictor",
    evaluation_strategy="steps",
    learning_rate=2e-5,
    per_device_train_batch_size=6,
    per_device_eval_batch_size=6,
    weight_decay=0.01,
    eval_steps=50,
    do_eval=True,
    save_total_limit=3,
    num_train_epochs=13,
    predict_with_generate=True,
    fp16=True,
    logging_dir='./logs',
    logging_strategy="steps",
    logging_steps=50,
    save_steps=500,
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

trainer.save_model("../model-backup/movie-overview-predictor")
