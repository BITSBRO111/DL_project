import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


def preprocess_sentiment_analysis():
    raw_file = os.path.join("DL_repo","data", "raw", "Sentiment Analysis", "Sentiment140.csv")
    processed_file = os.path.join("DL_repo","data", "processed", "sentiment_analysis_processed.csv")

    if not os.path.exists(raw_file):
        print(f"Error: File not found at {raw_file}")
        return

    print(f"Reading data from {raw_file}...")
    df = pd.read_csv(raw_file, encoding='latin-1', header=None)

    df.columns = ["target", "id", "date", "flag", "user", "text"]

    df = df[["target", "text"]]
    df["target"] = df["target"].apply(lambda x: 1 if x == 4 else 0) 

    os.makedirs(os.path.dirname(processed_file), exist_ok=True)
    print(f"Saving processed data to {processed_file}...")
    df.to_csv(processed_file, index=False)
    print("Processing complete.")


def preprocess_spam_detection():

    raw_file = os.path.join("DL_repo", "data", "raw", "Spam Detection", "Eron.csv")
    processed_file = os.path.join("DL_repo", "data", "processed", "spam_detection_processed.csv")

    if not os.path.exists(raw_file):
        print(f"Error: File not found at {raw_file}")
        return

    print(f"Reading data from {raw_file}...")
    df = pd.read_csv(raw_file, encoding='latin-1', header=None)

    df.columns = ["label", "text"]

    df['text'] = df['text'].str.lower()

    os.makedirs(os.path.dirname(processed_file), exist_ok=True)
    print(f"Saving processed data to {processed_file}...")
    df.to_csv(processed_file, index=False)
    print("Spam detection data preprocessing complete.")

def preprocess_topic_categorization():
    raw_folder = os.path.join("DL_repo", "data", "raw", "Topic Categorization")
    processed_file = os.path.join("DL_repo", "data", "processed", "topic_categorization_processed.csv")

    if not os.path.exists(raw_folder):
        print(f"Error: Folder not found at {raw_folder}")
        return

    texts = []
    for filename in os.listdir(raw_folder):
        if filename.endswith(".txt"):
            try:
                with open(os.path.join(raw_folder, filename), 'r', encoding='utf-8', errors='ignore') as file:
                    texts.append(file.read())
            except UnicodeDecodeError:
                print(f"Error decoding file {filename}, trying with ISO-8859-1...")
                with open(os.path.join(raw_folder, filename), 'r', encoding='ISO-8859-1', errors='ignore') as file:
                    texts.append(file.read())

    df = pd.DataFrame({'text': texts})

    df['text'] = df['text'].str.lower()

    os.makedirs(os.path.dirname(processed_file), exist_ok=True)
    print(f"Saving processed data to {processed_file}...")
    df.to_csv(processed_file, index=False)
    print("Topic categorization data preprocessing complete.")


if __name__ == "__main__":
    preprocess_sentiment_analysis()
    preprocess_spam_detection()
    preprocess_topic_categorization()



