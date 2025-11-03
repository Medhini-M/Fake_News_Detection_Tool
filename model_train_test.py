from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset, Value
from sklearn.metrics import accuracy_score, f1_score, classification_report
import pandas as pd
import torch

######## 1. Load and prepare data
df = pd.read_csv("train1.csv")[["text", "label"]]
test_df = pd.read_csv("test1.csv")[["text", "label"]]

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def tokenize(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

# Convert to Hugging Face Datasets
train_dataset = Dataset.from_pandas(df).map(tokenize, batched=True)
train_dataset = train_dataset.cast_column("label", Value("int64"))

test_dataset = Dataset.from_pandas(test_df).map(tokenize, batched=True)
test_dataset = test_dataset.cast_column("label", Value("int64"))


########## 2. Initialize model

num_labels = len(df["label"].unique())
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=num_labels)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


########## 3. Training arguments

training_args = TrainingArguments(
    output_dir="./my_roberta_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    save_safetensors=True,      
    learning_rate=2e-5,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.05,
    logging_dir="./logs",
    logging_steps=20,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    save_total_limit=1,
    report_to="none",
    warmup_steps=10,
    lr_scheduler_type="cosine",
)

def compute_metrics(p):
    preds = p.predictions.argmax(axis=1)
    labels = p.label_ids
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted")
    }

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

########### 4. Train and save model

print("Starting training...")
trainer.train()
trainer.save_model("./my_roberta_model")
tokenizer.save_pretrained("./my_roberta_model")
print("Model and tokenizer saved in ./my_roberta_model!!")


############ 5. Evaluate on test set

print("Evaluating on test data...")

# Reload to ensure loading from safetensors works
tokenizer = RobertaTokenizer.from_pretrained("./my_roberta_model")
model = RobertaForSequenceClassification.from_pretrained("./my_roberta_model")  # auto-loads model.safetensors

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Tokenize for inference
def tokenize_function(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

test_dataset = Dataset.from_dict({
    "text": test_df["text"].tolist(),
    "label": test_df["label"].tolist()
})
test_dataset = test_dataset.map(tokenize_function, batched=True)
test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

predictions = []
with torch.no_grad():
    for batch in torch.utils.data.DataLoader(test_dataset, batch_size=16):
        inputs = {k: v.to(device) for k, v in batch.items() if k != "label"}
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1)
        predictions.extend(preds.cpu().numpy())

acc = accuracy_score(test_df["label"], predictions)
f1 = f1_score(test_df["label"], predictions, average="weighted")
report = classification_report(test_df["label"], predictions)

print(f"\nAccuracy: {acc:.4f}")
print(f"F1 Score (weighted): {f1:.4f}")
print("\nClassification Report:")
print(report)


######### 6. Save predictions

output_df = pd.DataFrame({
    "text": test_df["text"],
    "true_label": test_df["label"],
    "predicted_label": predictions
})
output_df.to_csv("roberta_test_predictions.csv", index=False)
print("Predictions saved to 'roberta_test_predictions.csv'")

######### 7. User input

label_map = {0: "false", 1: "true"} 

def predict_text(input_text):
    inputs = tokenizer(
        input_text, return_tensors="pt", truncation=True,
        padding=True, max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=1)
        pred_id = torch.argmax(probs, dim=1).item()
        confidence = probs.max().item()

    return label_map[pred_id], confidence

user_input = input("\nEnter a news article: ")
prediction, conf = predict_text(user_input)
print(f"\nPrediction: {prediction.upper()} ({conf*100:.2f}% confidence)")
