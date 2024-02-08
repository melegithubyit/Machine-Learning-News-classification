import numpy as np
class TfidfVectorizer:
    def __init__(self):
        self.vocab = {}
        self.idf = {}

    def IDF_computer(self, data):
        total_documents = len(data)
        for document in data:
            words = set(document.split())
            for word in words:
                self.vocab.setdefault(word, 0)
                self.vocab[word] += 1

        for word, freq in self.vocab.items():
            self.idf[word] = np.log(total_documents / (freq + 1))

    def fit(self, data):
        self.IDF_computer(data)

    def transform(self, data):
        num_documents = len(data)
        features = np.zeros((num_documents, len(self.vocab)))
        for i, document in enumerate(data):
            words = document.split()
            word_counts = {word: words.count(word) for word in set(words)}
            for word, count in word_counts.items():
                if word in self.vocab:
                    features[i, self.vocab[word]] = count / len(words)
# tf-idf = count / len(words)

        features *= np.array(list(self.idf.values()))  
        return features

    def fit_transform(self, data):
        self.fit(data)
        return self.transform(data)

vectorizer = TfidfVectorizer()

def fit_vectorizer(data):
    vectorizer.fit(data['text'])

def extract_features(data):
    features = vectorizer.transform(data['text'])
    return features
