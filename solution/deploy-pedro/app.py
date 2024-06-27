from flask import Flask, request, render_template, jsonify
import pandas as pd
import tensorflow as tf
from transformers import pipeline, RobertaTokenizer, TFRobertaForSequenceClassification

app = Flask(__name__)

# Load the sentiment analysis model
model_path = "model/1"
model = TFRobertaForSequenceClassification.from_pretrained(model_path, local_files_only=True)
tokenizer = RobertaTokenizer.from_pretrained(model_path, local_files_only=True)

# Load the summarization pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Load the reviews CSV
data_df = pd.read_csv('data/data.csv')
summarized_reviews_df = pd.read_csv('data/summarized_reviews.csv')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/summarize', methods=['POST'])
def summarize():
    category = request.form.get('category')
    rating = request.form.get('rating')
    print(f"Received category: {category}, rating: {rating}")
    print("Columns in summarized_reviews_df:", summarized_reviews_df.columns)
    
    # Filter the DataFrame based on the selected category and rating
    filtered_reviews = summarized_reviews_df[
        (summarized_reviews_df['primaryCategories'] == category) & 
        (summarized_reviews_df['reviews.rating'] == int(rating))
    ]

    if filtered_reviews.empty:
        return jsonify({"summary": f"No {rating} star reviews for category {category}"})

    # Get the summary from the filtered reviews
    summary = " ".join(filtered_reviews['summary'].tolist())
    
    return jsonify({"summary": summary})

@app.route('/classify', methods=['POST'])
def classify():
    text = request.form.get('text')
    print(f"Received text: {text}")
    if not text:
        return jsonify({"error": "No text provided"})

    inputs = tokenizer(text, return_tensors="tf", padding=True, truncation=True, max_length=128)
    prediction = model(inputs)[0]
    sentiment = 'positive' if tf.argmax(prediction, axis=1).numpy()[0] == 1 else 'negative' if tf.argmax(prediction, axis=1).numpy()[0] == 0 else 'neutral'

    return jsonify({"sentiment": sentiment})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
