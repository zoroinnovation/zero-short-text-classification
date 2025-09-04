# Zero-Shot Text Classification using Prompt-Based Large Language Models



## 🧠 Overview
This project explores how large language models (LLMs) such as **T5, FLAN-T5, GPT-3.5, and Mistral** can be used for **zero-shot and few-shot text classification**.  
Instead of traditional fine-tuning, we rely on **natural language prompts** to guide the models.  
We also compare prompt-based methods with **fine-tuned BERT baselines**.

---

## 🎯 Objectives
- Understand zero-shot and few-shot learning
- Design and evaluate effective prompt templates
- Compare performance against supervised BERT-style models
- Test prompt generalization across domains (news, sentiment, questions, etc.)
- Deliver a structured and reusable pipeline for prompt engineering


---

## 📚 Datasets
- **SST-2 (Stanford Sentiment Treebank v2)**  
  - Binary sentiment classification: **Positive / Negative**  

- **AG News**  
  - News categorization into **4 classes**: World, Sports, Business, Sci/Tech  

---

## 🔧 Tech Stack
- **Python**
- **HuggingFace Transformers**
- **PyTorch**
- **Scikit-learn**
- **Matplotlib**

Models used:
- **Fine-Tuned BERT** (baseline supervised model)  
- **FLAN-T5** (zero-shot & few-shot prompting)

---

## 📊 Methodology
1. **Zero-Shot Prompting**
   - Created multiple prompt templates per dataset  
   - Example (SST-2):  
     ```
     Classify the sentiment of the following sentence strictly as Positive or Negative.
     Sentence: "This movie was a masterpiece of storytelling and emotion."
     Answer:
     ```
   - Example (AG News):  
     ```
     Classify the news article into one of these categories: World, Sports, Business, Sci/Tech.
     Article: "The stock market surged as new policies were announced."
     Answer:
     ```

2. **Few-Shot Prompting**
   - Added 2–5 labeled examples to prompts  
   - Balanced examples to avoid bias  
   - Tested for improvements in accuracy and generalization  

3. **Baseline Training**
   - Fine-tuned BERT separately on **SST-2** and **AG News**  

4. **Evaluation**
   - Metrics: Accuracy, Precision, Recall, F1  
   - Visualizations: Confusion matrices, comparison charts  

---

## ✅ Results
- **SST-2**
  - Fine-tuned BERT: Accuracy ~0.92, F1 ~0.93  
  - FLAN-T5 Zero-Shot: Dependent on prompt design  
  - FLAN-T5 Few-Shot: Improved recall & F1, closer to BERT  

- **AG News**
  - Fine-tuned BERT: Strong benchmark performance (Accuracy ~91.2, F1 ~90.8)  
  - FLAN-T5 Zero-Shot: Achieved good accuracy, but struggled with ambiguous news articles  
  - FLAN-T5 Few-Shot: Significant improvement compared to zero-shot, reduced errors  

**Key Takeaway:**  
👉 Prompt engineering is crucial — **clear, balanced prompts** significantly boost zero-shot/few-shot performance.  
👉 With few-shot prompting, FLAN-T5 narrowed the gap with fine-tuned BERT, especially on **SST-2 sentiment tasks**.  

---

## 📂 Repo Structure


.
├── codes (either ipynb or py)/
│   ├── ag_news.ipynb
│   ├── comparison_sst2.py
│   ├── load_sst2_dataset.ipynb
│   ├── sst2_bert.ipynb
│   ├── sst2_prompt_test.ipynb
│   └── test_sst2prompts.ipynb
│
├── data_csv/
│   ├── sst2_train_cleaned.csv
│   ├── sst2_train.csv
│   ├── sst2_val_cleaned.csv
│   └── sst2_val.csv
│
├── docs/
│   └── installation_guide.md
│
├── leaderboard/
│   └── leaderboard.csv
│
├── outputs/
│   ├── ag_news/
│   ├── sst2/
│   └── sst2_flan_predictions.json
│
├── prompts/
│   ├── ag_news/
│   ├── sst2/
│   └── trec/
│
├── venv/                # Virtual environment (not needed in GitHub)
│
├── .gitignore
├── README.md
├── requirements.txt
├── test_hf_api.ipynb
└── test_openai_api.ipynb


---

## 🚀 How to Run
1. Clone the repo:
   
   git clone https://github.com/zoroinnovation/zero-short-text-classification.git
   cd prompt-engineering-classification



2. Install dependencies:

pip install -r requirements.txt