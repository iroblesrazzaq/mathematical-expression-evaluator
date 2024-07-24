from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import base64
import cv2
import numpy as np
from io import BytesIO
from PIL import Image
import os
from backend.complete_model import (
    process_image, find_bb_contour, predict_elements, load_model,
    predictions_to_string, parse_expression, evaluate_expression,
    detect_equals_sign
)

app = Flask(__name__)
CORS(app, resources={r"/evaluate": {"origins": ["http://127.0.0.1:5500"]}}, 
     allow_headers=["Content-Type"], 
     supports_credentials=True)
logging.basicConfig(level=logging.INFO)  

model = load_model()
symbols_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'add', 'dec', 'div', 'eq', 'mul', 'sub', 'x', 'y', 'z']

def img_to_base64(img):
    img_pil = Image.fromarray(img)
    buffered = BytesIO()
    img_pil.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def handle_preflight():
    response = app.make_default_options_response()
    response.headers['Access-Control-Allow-Origin'] = 'http://127.0.0.1:5500'
    response.headers['Access-Control-Allow-Methods'] = 'POST'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
    response.headers['Access-Control-Allow-Credentials'] = 'true'
    return response

def process_and_predict(image_data):
    processed_img_np = process_image(image_data)
    cv2.imwrite('debug_processed_image.png', processed_img_np)
    #app.logger.debug(f"Processed image shape: {processed_img_np.shape}")
    
    bounding_boxes = find_bb_contour(processed_img_np)
    #app.logger.debug(f"Bounding boxes: {bounding_boxes}")
    
    predictions = predict_elements(processed_img_np, bounding_boxes, model)
    return processed_img_np, bounding_boxes, predictions

def draw_bounding_boxes(image, bounding_boxes):
    img_with_boxes = image.copy()
    height, width = img_with_boxes.shape[:2]
    #app.logger.debug(f"Image dimensions: {width}x{height}")
    
    if len(img_with_boxes.shape) == 2:  # If the image is grayscale
        img_with_boxes = cv2.cvtColor(img_with_boxes, cv2.COLOR_GRAY2BGR)
    
    for i, bbox in enumerate(bounding_boxes):
        x, y, w, h = bbox
        app.logger.debug(f"Drawing box {i}: x={x}, y={y}, w={w}, h={h}")
        
        # Check if the coordinates are within the image boundaries
        if x < 0 or y < 0 or x + w > width or y + h > height:
            app.logger.warning(f"Bounding box {i} is out of image boundaries!")
            continue
        
        cv2.rectangle(img_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    cv2.imwrite('debug_image_with_boxes.png', img_with_boxes)
    #app.logger.debug(f"Saved image with {len(bounding_boxes)} bounding boxes")
    return img_with_boxes   

def prepare_preprocessing_data(processed_img, bounding_boxes, predictions):
    img_with_boxes = draw_bounding_boxes(processed_img, bounding_boxes)
    preprocessing_data = {
        'processed_image': f"data:image/png;base64,{img_to_base64(img_with_boxes)}",
        'predictions': []
    }
    for (bbox, (pred_class, confidence)) in zip(bounding_boxes, predictions):
        x, y, w, h = bbox
        element = processed_img[y:y+h, x:x+w]
        preprocessing_data['predictions'].append({
            'image': f"data:image/png;base64,{img_to_base64(element)}",
            'symbol': symbols_list[pred_class],
            'confidence': confidence
        })
    return preprocessing_data

@app.route('/evaluate', methods=['POST', 'OPTIONS'])
def evaluate():
    if request.method == 'OPTIONS':
        return handle_preflight()

    #app.logger.info(f"Received request: {request.method}")
    #app.logger.info(f"Headers: {request.headers}")
    
    try:
        image_data = request.json['image']
        #app.logger.info("Received image data")
        
        processed_img_np, bounding_boxes, predictions = process_and_predict(image_data)
        
        #app.logger.info(f"Processed image shape: {processed_img_np.shape}")
        #app.logger.info(f"Number of bounding boxes: {len(bounding_boxes)}")
        #app.logger.info(f"Bounding boxes: {bounding_boxes}")
        
        predictions = predict_elements(processed_img_np, bounding_boxes, model)
        #app.logger.info(f"Predictions: {predictions}")
        
        expression_with_bbox = predictions_to_string(predictions, bounding_boxes, symbols_list)
        #app.logger.info(f"Expression with bounding boxes: {expression_with_bbox}")

        expression_with_equals = detect_equals_sign(expression_with_bbox)
        #app.logger.info(f"Expression with equals detection: {expression_with_equals}")

        expression_string = ''.join([symbol for symbol, _ in expression_with_equals])
        #app.logger.info(f"Final expression string: {expression_string}")

        parsed_expression = parse_expression(expression_string)
        #app.logger.info(f"Parsed expression: {parsed_expression}")
        
        result = evaluate_expression(parsed_expression)
        #app.logger.info(f"Evaluation result: {result}")
        
        preprocessing_data = prepare_preprocessing_data(processed_img_np, bounding_boxes, predictions)
        
        return jsonify({
            'expression': parsed_expression,
            'result': str(result),
            'preprocessing': preprocessing_data
        })
    except Exception as e:
        app.logger.error(f"Error in evaluate: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False, use_reloader=False, threaded=True)