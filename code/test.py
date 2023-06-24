from transformers import (
    AutoTokenizer,
    LongT5Model,
    pipeline
)
import pandas as pd
from datasets import DatasetDict, Dataset

model_name = "google/long-t5-tglobal-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = LongT5Model.from_pretrained(model_name)

df = pd.read_csv('../data/movies_preprocessed.csv')

movies = Dataset.from_pandas(df)

subtitle = df['subtitles'][0]

summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)
print("A")
print(summarizer(subtitle))
print("A1")


"""
inputs = tokenizer(subtitle, return_tensors="pt")
print("A1")
outputs = model(**inputs)
print(outputs)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
print("uasa")
"""