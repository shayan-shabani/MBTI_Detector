import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification

dataset = pd.read_csv("/Users/shayanshabani/Downloads/twitter_MBTI.csv")
dataset = dataset[['text', 'label']]
label_map = {"intj": 0, "intp": 1, "entj": 2, "entp": 3, "infj": 4, "infp": 5, "enfj": 6, "enfp": 7,
             "istj": 8, "isfj": 9, "estj": 10, "esfj": 11, "istp": 12, "isfp": 13, "estp": 14, "esfp": 15}
dataset["label"] = dataset["label"].apply(lambda x: label_map[x])
dataset = dataset[0:7000]

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=16)

X = list(dataset["text"])
y = list(dataset["label"])
X_train, X_val, Y_train, Y_val = train_test_split(X, y, test_size=0.2, stratify=y)
X_train_tokenized = tokenizer(X_train, padding=True, truncation=True, max_length=512)
X_val_tokenized = tokenizer(X_val, padding=True, truncation=True, max_length=512)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings['input_ids'])


training_dataset = Dataset(X_train_tokenized, Y_train)
validation_dataset = Dataset(X_val_tokenized, Y_val)


def compute_metrics(p):
    print(type(p))
    pred, labels = p
    pred = np.argmax(pred, axis=1)
    accuracy = accuracy_score(y_true=labels, y_pred=pred)
    f1 = f1_score(y_true=labels, y_pred=pred, average="micro")
    return {"accuracy": accuracy, "f1": f1}


args = TrainingArguments(
    output_dir="output",
    num_train_epochs=1,
    per_device_train_batch_size=8
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=training_dataset,
    eval_dataset=validation_dataset,
    compute_metrics=compute_metrics
)

trainer.train()
trainer.evaluate()
trainer.save_model('CustomModel')