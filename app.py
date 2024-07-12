from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from model_scripts.complete_model import process_image, find_bb_cca, \
find_bb_contour, predict_elements, load_model, predictions_to_string, parse_expression, evaluate_expression

app = Flask(__name__)
CORS(app, resources={r"/evaluate": {"origins": "http://127.0.0.1:5500"}}, allow_headers=["Content-Type"], supports_credentials=True)

logging.basicConfig(level=logging.DEBUG)

model = load_model()  # Load your model here
symbols_list = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'add', 'dec', 'div', 'eq', 'mul', 'sub', 'x', 'y', 'z']

@app.route('/evaluate', methods=['POST', 'OPTIONS'])
def evaluate():
    if request.method == 'OPTIONS':
        # Handling preflight request
        response = app.make_default_options_response()
        response.headers['Access-Control-Allow-Origin'] = 'http://127.0.0.1:5500'
        response.headers['Access-Control-Allow-Methods'] = 'POST'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        response.headers['Access-Control-Allow-Credentials'] = 'true'
        return response

    app.logger.info(f"Received request: {request.method}")
    app.logger.info(f"Headers: {request.headers}")
    
    try:
        # Get the image data from the request
        image_data = request.json['image']
        app.logger.info("Received image data")
        
        # Process the image        
        processed_img_np = process_image(image_data)

        # find bounding boxes
        bounding_boxes = find_bb_contour(processed_img_np) # contour analysis
        bounding_boxes = find_bb_cca(processed_img_np) # connected component analysis

        #app.logger.info(f"Processed image shape: {processed_image.shape}")
        #app.logger.info(f"Number of bounding boxes: {len(bounding_boxes)}")
        #app.logger.info(f"Bounding boxes: {bounding_boxes}")
        
        # Make predictions
        predictions = predict_elements(processed_img_np, bounding_boxes, model)
        app.logger.info(f"Predictions: {predictions}")
        
        # Convert predictions to a string representation
        expression_string = predictions_to_string(predictions, symbols_list)
        #app.logger.info(f"Expression string: {expression_string}")

        # Parse and evaluate the expression
        parsed_expression = parse_expression(expression_string)
        app.logger.info(f"Parsed expression: {parsed_expression}")
        result = evaluate_expression(parsed_expression)
        app.logger.info(f"Evaluation result: {result}")
        
        # Return the results
        return jsonify({
            'expression': parsed_expression,
            'result': str(result)
        })
    except Exception as e:
        app.logger.error(f"Error in evaluate: {str(e)}", exc_info=True)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)