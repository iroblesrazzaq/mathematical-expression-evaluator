from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from model_scripts.complete_model import process_image, find_bb_cca, \
find_bb_contour, predict_elements, load_model, predictions_to_string, \
parse_expression, evaluate_expression, classify_symbol, post_process_expression

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
        # Find bounding boxes using the improved contour method
        bounding_boxes = find_bb_contour(processed_img_np)
        # bounding_boxes = find_bb_cca(processed_img_np)

        app.logger.info(f"Processed image shape: {processed_img_np.shape}")
        app.logger.info(f"Number of bounding boxes: {len(bounding_boxes)}")
        app.logger.info(f"Bounding boxes: {bounding_boxes}")
        
        # Classify symbols based on bounding box characteristics
        symbol_classifications = [classify_symbol(processed_img_np, box) for box in bounding_boxes]
        app.logger.info(f"Symbol classifications: {symbol_classifications}")

        # Make predictions using the model
        predictions = predict_elements(processed_img_np, bounding_boxes, model)
        app.logger.info(f"Model predictions: {predictions}")
        
        # Combine model predictions with symbol classifications
        combined_predictions = []
        for (pred_class, confidence), symbol_class, box in zip(predictions, symbol_classifications, bounding_boxes):
            if symbol_class in ['eq', 'sub'] and confidence < 0.9:
                app.logger.info(f"Overriding model prediction {symbols_list[pred_class]} ({confidence:.2f}) with {symbol_class} for box {box}")
                combined_predictions.append((symbols_list.index(symbol_class), 0.9))
            else:
                combined_predictions.append((pred_class, confidence))
        
        # Convert predictions to a string representation
        expression_string = predictions_to_string(combined_predictions, symbols_list)
        app.logger.info(f"Initial expression string: {expression_string}")

        # Post-process the expression
        processed_expression = post_process_expression(expression_string)
        app.logger.info(f"Processed expression string: {processed_expression}")

        # Parse and evaluate the processed expression
        parsed_expression = parse_expression(processed_expression)
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