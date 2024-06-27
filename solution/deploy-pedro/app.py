from flask import Flask, request, render_template, jsonify
import pandas as pd
import tensorflow as tf
from transformers import pipeline, DistilBertTokenizer, TFDistilBertForSequenceClassification

app = Flask(__name__)

# Load the sentiment analysis model
model_path = "model/1"  # Adjust the path to your model
model = TFDistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizer.from_pretrained(model_path)

# Load the summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Load the reviews CSV
reviews_df = pd.read_csv('data/reviews.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    category = request.form['category']
    rating = request.form['rating']
    filtered_reviews = reviews_df[(reviews_df['category'] == category) & (reviews_df['rating'] == int(rating))]

    if filtered_reviews.empty:
        return jsonify({"summary": f"No {rating} star reviews for category {category}"})

    # Summarize the reviews
    texts = filtered_reviews['review_text'].tolist()
    summaries = summarizer(texts, max_length=50, min_length=25, do_sample=False)
    summarized_text = " ".join([summary['summary_text'] for summary in summaries])
    return jsonify({"summary": summarized_text})

@app.route('/classify', methods=['POST'])
def classify():
    text = request.form['text']
    if not text:
        return jsonify({"error": "No text provided"})

    # Tokenize and classify text
    inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True, max_length=128)
    prediction = model(inputs)[0]
    sentiment = 'positive' if tf.argmax(prediction, axis=1).numpy()[0] == 1 else 'negative' if tf.argmax(prediction, axis=1).numpy()[0] == 0 else 'neutral'

    return jsonify({"sentiment": sentiment})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
