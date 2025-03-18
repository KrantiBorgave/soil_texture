from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image
import io
from model import predict_soil

app = Flask(__name__)
CORS(app)  # Important for cross-origin requests

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    try:
        # Read image
        file = request.files['image']
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        img_array = np.array(img)
        
        # Get prediction
        soil_type = predict_soil(img_array)
        return jsonify({'soil_type': soil_type})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)