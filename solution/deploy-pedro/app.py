from flask import Flask, request, render_template, jsonify
import pandas as pd
import tensorflow as tf
from transformers import pipeline

app = Flask(__name__)

# Load the sentiment analysis model
model_path = "model/1"  # Adjust the path to your model
model = tf.keras.models.load_model(model_path)
tokenizer = tf.keras.preprocessing.text.Tokenizer()

# Load the summarization pipeline (e.g., BART)
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
    tokens = tokenizer.texts_to_sequences([text])
    padded_tokens = tf.keras.preprocessing.sequence.pad_sequences(tokens, maxlen=128)
    prediction = model.predict(padded_tokens)
    sentiment = 'positive' if prediction > 0.5 else 'negative' if prediction < 0.5 else 'neutral'

    return jsonify({"sentiment": sentiment})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
