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

df = pd.read_csv('./dataset/dataset_True_True_True_True_True.csv')
test_split = 0.3

movies = Dataset.from_pandas(df)

tokenizer = AutoTokenizer.from_pretrained("t5-small")

def split_subtitle(subtitle, max_length, window_size):
    segments = []
    start = 0
    while start < len(subtitle):
        end = min(start + max_length, len(subtitle))
        segment = subtitle[start:end]
        segments.append(segment)
        start += window_size
    return segments

def preprocess(examples):
    t5_task_prefix = "summarize: " 
    inputs = [t5_task_prefix + doc for doc in examples["subtitles"]]

    # Split subtitle into segments using sliding window approach
    max_length = 512  # Maximum sequence length supported by the model
    window_size = 256  # Sliding window size

    segmented_inputs = []
    for input_text in inputs:
        segments = split_subtitle(input_text, max_length, window_size)
        segmented_inputs.extend(segments)

    # Tokenize segmented inputs
    model_inputs = tokenizer(segmented_inputs, max_length=1024, truncation=True)

    # Prepare labels
    labels = tokenizer(text_target=examples["overview"], max_length=128, truncation=True)

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# Apply the sliding window approach to split subtitle strings
tokenized_subtitles = movies.map(preprocess, batched=True, num_proc=4, remove_columns=[])

train_test = tokenized_subtitles.train_test_split(shuffle=True, test_size=test_split, seed=1)

tokenized_subtitles = DatasetDict(
    train=train_test["train"],
    test=train_test["test"],
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

model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

training_args = Seq2SeqTrainingArguments(
    output_dir="movie-overview-predictor",
    evaluation_strategy="steps",
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    eval_steps=50,
    do_eval=True,
    save_total_limit=3,
    num_train_epochs=10,
    predict_with_generate=True,
    fp16=True,
    logging_dir='./logs',
    logging_strategy="steps",
    logging_steps=50,
    save_steps=500,
    push_to_hub=False,
    warmup_steps=500  # Adjust this value based on the total number of training steps
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

trainer.save_model("movie-overview-predictor")