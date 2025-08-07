from transformers import T5Tokenizer, TFT5ForConditionalGeneration

model_name = "google/flan-t5-base"
tokenizer = T5Tokenizer.from_pretrained(model_name)
model = TFT5ForConditionalGeneration.from_pretrained(model_name)

text = "Classify the sentiment of this sentence: I love this product!"
input_ids = tokenizer(text, return_tensors="tf").input_ids
outputs = model.generate(input_ids)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
