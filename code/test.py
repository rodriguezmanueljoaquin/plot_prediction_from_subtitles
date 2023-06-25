from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline
)
import pandas as pd
from datasets import DatasetDict, Dataset
import torch
model_name = "t5-small"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda")

input("hola")

df = pd.read_csv('../data/dataset_True_True_True_True_True.csv')

movies = Dataset.from_pandas(df)

subtitle = df['subtitles'][1]

summarizer = pipeline("summarization", model=model, tokenizer=tokenizer,device=0)
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