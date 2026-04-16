from transformers import T5Tokenizer, T5ForConditionalGeneration
import torch

model_name = "t5-large" 
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)

input_text = "translate English to Sinhala: How are you?"
inputs = tokenizer(input_text, return_tensors="pt")
translation_ids = model.generate(inputs.input_ids, max_length=50, num_beams=5, early_stopping=True)
translation_text = tokenizer.decode(translation_ids[0], skip_special_tokens=True)
print("Translation:", translation_text)
