import pandas as pd
import string

def load_data(filename):
    data = pd.read_csv(filename)
    data['text'] = data['text'].apply(clean_text)
    return data

def clean_text(text):
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text