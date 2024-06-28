import os
import pandas as pd
import tensorflow as tf
from transformers import TFRobertaForSequenceClassification, RobertaTokenizer
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

# Load the pre-trained model and tokenizer
model_path = "model/1"
model = TFRobertaForSequenceClassification.from_pretrained(model_path)
tokenizer = RobertaTokenizer.from_pretrained(model_path)

# Load summarized reviews data
summarized_reviews_df = pd.read_csv('data/summarized_reviews.csv')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    category = request.form.get('category')
    rating = request.form.get('rating')
    filtered_reviews = summarized_reviews_df[
        (summarized_reviews_df['primaryCategories'] == category) &
        (summarized_reviews_df['reviews.rating'] == int(rating))
    ]

    if filtered_reviews.empty:
        return jsonify({"summary": f"No {rating} star reviews for category {category}"})

    summary = filtered_reviews['summary'].tolist()[0]
    return jsonify({"summary": summary})

@app.route('/classify', methods=['POST'])
def classify():
    text = request.form.get('text')
    inputs = tokenizer(text, return_tensors="tf", truncation=True, padding=True)
    prediction = model(inputs)[0]
    sentiment = 'positive' if tf.argmax(prediction, axis=1).numpy()[0] == 1 else 'negative' if tf.argmax(prediction, axis=1).numpy()[0] == 0 else 'neutral'
    return jsonify({"sentiment": sentiment})

# Run the app
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=os.getenv("PORT", default=5000))
