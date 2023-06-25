from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback
)

generator_model_name = "facebook/bart-base"
model = AutoModelForSeq2SeqLM.from_pretrained(generator_model_name)
print(model)
