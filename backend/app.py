from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import io
import re
import joblib
from PIL import Image
import pytesseract
from utils import clean_text

app = Flask(__name__)
CORS(app) 

model = joblib.load('model/sentiment_model.pkl')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'text' in request.form:
        text = clean_text(request.form['text'])
    elif 'image' in request.files:
        img = Image.open(request.files['image'])
        text = pytesseract.image_to_string(img)
        text = clean_text(text)
    else:
        return jsonify({'error': 'No valid input provided.'}), 400

    prediction = model.predict([text])[0]
    sentiment = 'Positive' if prediction == 1 else 'Negative'
    return jsonify({'sentiment': sentiment, 'text': text})

if __name__ == '__main__':
    app.run(debug=True)
