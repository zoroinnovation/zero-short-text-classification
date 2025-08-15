import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pandas as pd
from tqdm import tqdm
import re
import os
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset

# -----------------------
# Load FLAN-T5 model and tokenizer
# -----------------------
model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Move model to device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Model loaded on {device}")

# -----------------------
# Load TREC dataset (replacement for deprecated `load_dataset("trec")`)
# -----------------------
dataset = load_dataset(
    "csv",
    data_files={
        "train": "https://raw.githubusercontent.com/whangbo/trec-dataset/master/train_5500.label",
        "test": "https://raw.githubusercontent.com/whangbo/trec-dataset/master/TREC_10.label"
    },
    delimiter="\t",
    column_names=["label", "text"]
)

# Extract coarse label from "DESC:manner" format
def split_labels(example):
    coarse = example["label"].split(":")[0]
    return {"coarse_label": coarse}

dataset = dataset.map(split_labels)

labels_map = {
    "DESC": "Description",
    "ENTY": "Entity",
    "ABBR": "Abbreviation",
    "HUM": "Human",
    "LOC": "Location",
    "NUM": "Numeric"
}

# -----------------------
# Load prompt template
# -----------------------
def load_prompt_template(path, template_number=1):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    
    pattern = rf"Template {template_number}:(.*?)(?:Template \d+:|$)"
    match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
    
    if not match:
        raise ValueError(f"❌ Template {template_number} not found")
    
    raw_prompt = match.group(1).strip()
    
    cleaned_lines = [
        line for line in raw_prompt.splitlines()
        if line.strip() and not line.strip().startswith("#") and not line.strip().startswith("=")
    ]
    
    return "\n".join(cleaned_lines).strip()

prompt_template = load_prompt_template(
    "../prompts/trec/trec_prompt_templates.txt",
    template_number=1
)
print("✅ Extracted Prompt Template:\n", prompt_template)

# -----------------------
# Helpers for prompt and prediction
# -----------------------
def generate_prompt(text):
    return prompt_template.replace("{question}", text)

def flan_predict(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)
    outputs = model.generate(**inputs, max_new_tokens=10)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# -----------------------
# Process test data (500 samples for speed)
# -----------------------
test_data = dataset["test"].select(range(500))
predictions = []

print("🚀 Starting predictions...")
for row in tqdm(test_data):
    text = row["text"]
    true_label = labels_map[row["coarse_label"]]
    
    prompt = generate_prompt(text)
    output = flan_predict(prompt)
    
    response = output.strip().lower()
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    if lines:
        response = lines[-1]
    response = re.sub(r"[.:]", "", response)
    
    pred = "Unknown"
    for code, name in labels_map.items():
        if name.lower() in response or code.lower() in response:
            pred = name
            break
    
    predictions.append({
        "text": text,
        "true_label": true_label,
        "prediction": pred
    })

# -----------------------
# Save predictions
# -----------------------
os.makedirs("../outputs/trec", exist_ok=True)
predictions_file = "../outputs/trec/trec_predictions.csv"
pd.DataFrame(predictions).to_csv(predictions_file, index=False)
print(f"✅ Predictions saved to {predictions_file}")

# -----------------------
# Metrics
# -----------------------
result_df = pd.DataFrame(predictions)
filtered_df = result_df[result_df["prediction"] != "Unknown"]

if len(filtered_df) > 0:
    acc = accuracy_score(filtered_df["true_label"], filtered_df["prediction"])
    precision, recall, f1, _ = precision_recall_fscore_support(
        filtered_df["true_label"], filtered_df["prediction"], 
        average="macro", zero_division=0
    )
    
    metrics_file = "../outputs/trec/metrics.csv"
    pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1 Score"],
        "Value": [acc, precision, recall, f1]
    }).to_csv(metrics_file, index=False)
    
    print(f"\n📊 Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"✅ Metrics saved to {metrics_file}")
    
    cm = confusion_matrix(filtered_df["true_label"], filtered_df["prediction"], labels=list(labels_map.values()))
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=list(labels_map.values()),
                yticklabels=list(labels_map.values()))
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - TREC (FLAN-T5)")
    confusion_matrix_file = "../outputs/trec/trec_confusion_matrix.png"
    plt.savefig(confusion_matrix_file)
    plt.show()
else:
    print("❌ No valid predictions found for metrics calculation")

print("\n🎉 TREC question classification completed!")
