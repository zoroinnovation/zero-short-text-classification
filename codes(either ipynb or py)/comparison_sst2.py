# Zero-Shot vs Few-Shot on SST-2 (FLAN-T5)




import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os, re


# 1. Load Model & Tokenizer

model_name = "google/flan-t5-large"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("✅ Model loaded on", device) 


# 2. Load SST-2 Validation Set

df = pd.read_csv(r"D:\Zoro_project2\zero-short-text-classification\data_csv\sst2_val_cleaned.csv").dropna(subset=["text"])
df["true_label"] = df["label"].map({1: "Positive", 0: "Negative"})


# 3. Load Prompt Templates

def load_template(path, template_number=1):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    # Regex to isolate template
    pattern = rf"Template {template_number}:(.*?)(?:Template \d+:|$)"
    match = re.search(pattern, content, re.DOTALL | re.IGNORECASE)
    if not match:
        raise ValueError(f"Template {template_number} not found in {path}")
    raw_prompt = match.group(1).strip()
    cleaned = "\n".join(
        line for line in raw_prompt.splitlines()
        if line.strip() and not line.strip().startswith("#") and not line.strip().startswith("=")
    )
    return cleaned.strip()

zero_template = load_template(r"D:\Zoro_project2\zero-short-text-classification\prompts\sst2\zero_shot_prompt.txt", template_number=1)
with open(r"D:\Zoro_project2\zero-short-text-classification\prompts\sst2\few_shot_prompt.txt", "r", encoding="utf-8") as f:
    few_template = f.read().strip()

# 4. Helpers

def generate_prompt(template, sentence):
    return template.replace("{sentence}", sentence)

def flan_predict(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=5)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def normalize_prediction(raw):
    raw = raw.strip().lower()
    if "positive" in raw:
        return "Positive"
    elif "negative" in raw:
        return "Negative"
    return "Unknown"


# 5. Evaluate Function

def evaluate(template, dataset, name="zero_shot"):
    predictions = []
    for _, row in tqdm(dataset.iterrows(), total=len(dataset), desc=f"Running {name}"):
        prompt = generate_prompt(template, row["text"])
        output = flan_predict(prompt)
        pred = normalize_prediction(output)
        predictions.append({
            "text": row["text"],
            "true_label": row["true_label"],
            "prediction": pred,
            "raw_output": output
        })

    result_df = pd.DataFrame(predictions)
    result_df.to_csv(r"D:\Zoro_project2\zero-short-text-classification\outputs\sst2/{name}_predictions.csv", index=False)

    filtered = result_df[result_df["prediction"] != "Unknown"]
    acc = accuracy_score(filtered["true_label"], filtered["prediction"])
    prec = precision_score(filtered["true_label"], filtered["prediction"], pos_label="Positive")
    rec = recall_score(filtered["true_label"], filtered["prediction"], pos_label="Positive")
    f1 = f1_score(filtered["true_label"], filtered["prediction"], pos_label="Positive")

    metrics = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
    with open(f"outputs/sst2/{name}_metrics.txt", "w") as f:
        f.write(str(metrics))

    print(f"\n📊 {name.upper()} Results: {metrics}")
    return metrics, result_df


# 6. Run Zero-Shot & Few-Shot

os.makedirs(r"D:\Zoro_project2\zero-short-text-classification\outputs\sst2", exist_ok=True)
zero_metrics, zero_preds = evaluate(zero_template, df, "zero_shot")
few_metrics, few_preds = evaluate(few_template, df, "few_shot")


# 7. Compare Metrics (Bar Plot)

metrics_df = pd.DataFrame([
    {"method": "Zero-Shot", **zero_metrics},
    {"method": "Few-Shot", **few_metrics}
])

metrics_df.set_index("method")[["accuracy", "precision", "recall", "f1"]].plot(
    kind="bar", figsize=(8,6), colormap="Set2", rot=0
)
plt.title("FLAN-T5 on SST-2: Zero-Shot vs Few-Shot")
plt.ylabel("Score")
plt.ylim(0, 1.0)
plt.tight_layout()
plt.savefig(r"D:\Zoro_project2\zero-short-text-classification\outputs\sst2\comparison_chart.png")
plt.show()


# 8. Confusion Matrices

def plot_confusion(df, name):
    cm = confusion_matrix(df["true_label"], df["prediction"], labels=["Positive","Negative"])
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Positive","Negative"],
                yticklabels=["Positive","Negative"])
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(r"D:\Zoro_project2\zero-short-text-classification\outputs\sst2/{name}_confusion_matrix.png")
    plt.show()

plot_confusion(zero_preds, "Zero-Shot")
plot_confusion(few_preds, "Few-Shot")

