from flask import Flask, render_template, jsonify, request
from pydantic import BaseModel
from typing import List, Union
import json

from attributor.evaluation.evaluation_case import EvaluationCase, EvaluationResult

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_file', methods=['POST'])
def process_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        try:
            data = json.load(file)
            results = [EvaluationResult(**result) for result in data]
            return jsonify([result.dict() for result in results])
        except json.JSONDecodeError:
            return jsonify({'error': 'Invalid JSON file'})
if __name__ == '__main__':
    app.run(debug=True)