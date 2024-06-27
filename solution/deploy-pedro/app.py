from flask import Flask, request, render_template, jsonify
import pandas as pd
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
    category = request.form['category']
    rating = request.form['rating']
    filtered_reviews = summarized_reviews_df[(summarized_reviews_df['category'] == category) & (summarized_reviews_df['rating'] == int(rating))]

    if filtered_reviews.empty:
        return jsonify({"summary": f"No {rating} star reviews for category {category}"})

    # Return the pre-summarized reviews
    summarized_text = " ".join(filtered_reviews['summary'].tolist())
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