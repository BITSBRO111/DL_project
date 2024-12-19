import os
import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        text = str(self.texts[item])
        encoding = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[item], dtype=torch.long)
        }

def create_dataset(df, tokenizer, max_len):
    dataset = TextDataset(
        texts=df.text.to_numpy(),
        labels=df.target.to_numpy(),
        tokenizer=tokenizer,
        max_len=max_len
    )
    return dataset

def train_model(df, model_name, num_classes, max_len, batch_size, epochs, learning_rate):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    # Split the dataset into training and validation
    train_df, val_df = df[:int(0.9*len(df))], df[int(0.9*len(df)):]

    train_dataset = create_dataset(train_df, tokenizer, max_len)
    val_dataset = create_dataset(val_df, tokenizer, max_len)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_classes)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        evaluation_strategy='epoch',
        learning_rate=learning_rate,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=10,
        save_steps=1000,
        save_total_limit=2,  # Only save the last 2 checkpoints
        eval_steps=500  # Interval for evaluation
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset  # Provide the validation dataset
    )

    trainer.train()
    model.save_pretrained(f"./models/{model_name}_{num_classes}.bin")

def evaluate_model(df, model_name, max_len, batch_size):
    tokenizer = BertTokenizer.from_pretrained(model_name)
    test_dataset = create_dataset(df, tokenizer, max_len)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = BertForSequenceClassification.from_pretrained(model_name)
    model.eval()

    predictions, true_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions.extend(torch.argmax(logits, dim=1).tolist())
            true_labels.extend(labels.tolist())

    return pd.DataFrame({
        'true_label': true_labels,
        'predicted_label': predictions
    })

def train_sentiment_analysis(df):
    model_name = 'bert-base-uncased'
    num_classes = 2
    max_len = 160
    batch_size = 32
    epochs = 4
    learning_rate = 2e-5

    train_model(df, model_name, num_classes, max_len, batch_size, epochs, learning_rate)

def train_spam_detection(df):
    model_name = 'bert-base-uncased'
    num_classes = 2
    max_len = 160
    batch_size = 32
    epochs = 4
    learning_rate = 2e-5

    train_model(df, model_name, num_classes, max_len, batch_size, epochs, learning_rate)

def train_topic_categorization(df):
    model_name = 'bert-base-uncased'
    num_classes = len(df['text'].unique())  # Assuming one class per unique text
    max_len = 160
    batch_size = 32
    epochs = 4
    learning_rate = 2e-5

    train_model(df, model_name, num_classes, max_len, batch_size, epochs, learning_rate)

if __name__ == '__main__':
    # Sentiment Analysis
    sentiment_df = pd.read_csv("DL_repo/data/processed/sentiment_analysis_processed.csv")
    train_sentiment_analysis(sentiment_df)

    # Spam Detection
    spam_df = pd.read_csv("DL_repo/data/processed/spam_detection_processed.csv")
    train_spam_detection(spam_df)

    # Topic Categorization
    topic_df = pd.read_csv("DL_repo/data/processed/topic_categorization_processed.csv")
    train_topic_categorization(topic_df)
