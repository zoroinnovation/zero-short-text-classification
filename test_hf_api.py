from huggingface_hub import InferenceClient

# Replace with your actual Hugging Face API token
api_token = "your_api_key"

client = InferenceClient(token=api_token)

def classify_sentiment(text):
    response = client.text_classification(text)
    # Pick the label with the highest score
    best_label = max(response, key=lambda x: x.score).label
    return best_label

texts = [
    "I love this product!",
    "The service was terrible.",
    "It is okay, nothing special."
]

for text in texts:
    result = classify_sentiment(text)
    print(f'Text: "{text}" -> Sentiment: {result}')
