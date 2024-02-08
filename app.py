from flask import Flask, render_template, request, flash
from data_loader import load_data
from feature_extractor import extract_features
from evaluator import evaluate
from model import train_model, predict
import pandas as pd
from feature_extractor import fit_vectorizer
import secrets
from collections import Counter

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)
model = None
test_data = None
test_features = None
test_labels = None

data = load_data('train.csv')
fit_vectorizer(data)

features = extract_features(data)
labels = data['category']

# print(Counter(labels))

model = train_model(features, labels)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/proceed")
def start():
    return render_template("options.html")

@app.route("/predicting")
def predicting():
    return render_template("index.html")

@app.route("/testing")
def testing():
    return render_template("test.html", accuracy=None)

@app.route('/predict', methods=['POST'])
def predict_category():
    global model
    news = request.form['news']
    news_features = extract_features(pd.DataFrame({'text': [news]}))
    prediction = predict(model, news_features)[0]
    return render_template('index.html', prediction=prediction)

@app.route('/test', methods=['GET', 'POST'])
def test():
    accuracy = None
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file uploaded')
        else:
            file = request.files['file']
            if file.filename == '':
                flash('No file selected')
            else:
                test_data = load_data(file)
                test_features = extract_features(test_data)
                test_labels = test_data['category']
                predictions = predict(model, test_features)
                accuracy = evaluate(predictions, test_labels)
                flash(f'Test accuracy: {accuracy:.2f}%')
    return render_template('test.html', accuracy=accuracy)

if __name__ == '__main__':
    app.run(debug=True)