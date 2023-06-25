from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainerCallback
)

# model_name = "/opt/movies/runs/t5-base_lr2e-05_epochs100/model"
# model_name = "/opt/movies/runs/facebook_bart-base_lr2e-05_epochs100/model"
model_name = "/opt/movies/runs/google_pegasus-large_lr2e-05_epochs50/train-results/checkpoint-3500"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name, return_dict=True).to('cuda')
overview = "Half human, half Atlantean, Arthur Curry is an inhabitant of the powerful underwater kingdom of Atlantis raised by a human man and considered an outcast by his own kind. Arthur will embark on a journey that will help him discover if he is worthy of fulfilling his destiny: to be king"
text = f'summarize: {overview}'
inputs = tokenizer.encode(text, return_tensors="pt", truncation=True).to('cuda')
outputs = model.generate(inputs, do_sample=False, num_beams=3, max_length=10, early_stopping=True)
output = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(f'predicted: {output}')

