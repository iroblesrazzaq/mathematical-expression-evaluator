from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
from PIL import Image
import numpy as np
import cv2
import torch
from model_scripts.cnn_class import CNN  # Import your CNN class
from model_scripts.segmentaion_model_2 import process_and_segment_image, 

app = Flask(__name__)
CORS(app)  # This allows your frontend to make requests to this server

# Load your model
model = CNN(num_classes=19)  # Adjust num_classes if needed
model.load_state_dict(torch.load("./model_scripts/normalized_model.pth"))
model.eval()

@app.route('/evaluate', methods=['POST'])
def evaluate():
    # Get the image data from the request
    image_data = request.json['image']
    
    # Process the image
    processed_image, bounding_boxes = process_and_segment_image(image_data)
    
    # Make predictions
    predictions = predict_elements(processed_image, bounding_boxes, model)
    
    # Convert predictions to a string representation
    expression_string = predictions_to_string(predictions, symbols_list)
    
    # Parse and evaluate the expression
    parsed_expression = parse_expression(expression_string)
    result = evaluate_expression(parsed_expression)
    
    # Return the results
    return jsonify({
        'expression': expression_string,
        'result': str(result)
    })

if __name__ == '__main__':
    app.run(debug=True)