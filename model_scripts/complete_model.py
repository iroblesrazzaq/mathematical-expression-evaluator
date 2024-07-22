
import numpy as np
import cv2
import tkinter as tk
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from model_scripts.cnn_class import CNN
import matplotlib.pyplot as plt
import os
import random
import re
import math
from sympy import symbols, parse_expr 
import base64
import io
import logging

def process_image(image_data, target_width=400):
    # Decode the base64 image
    debug_dir = 'debug_images'
    os.makedirs(debug_dir, exist_ok=True)

    image_data = image_data.split(',')[1]  # Remove the "data:image/png;base64," part
    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    
    # Convert transparent background to white
    if image.mode in ('RGBA', 'LA') or (image.mode == 'P' and 'transparency' in image.info):
        alpha = image.convert('RGBA').split()[-1]
        bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
        bg.paste(image, mask=alpha)
        image = bg.convert('RGB')
    
    # Save the white background image for inspection
    # image.save('debug_images/white_background_image.png')
    
    # Convert to numpy array
    image_np = np.array(image)
    
    # Convert to grayscale
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    image_np = cv2.bitwise_not(image_np)
    
    # Resize to target size while maintaining aspect ratio
    target_height = target_width // 2
    image_np = cv2.resize(image_np, (target_width, target_height), interpolation=cv2.INTER_AREA)
    cv2.imwrite('debug_images/resized_image.png', image_np)

    return image_np


# Bounding box finding image with Connected Component Analysis (CCA)
def find_bb_contour(image_np, min_ratio=0.0005, max_ratio=0.25, merge_distance=10):
    binary = cv2.adaptiveThreshold(image_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
    
    image_area = image_np.shape[0] * image_np.shape[1]
    filtered_boxes = []
    for box in bounding_boxes:
        x, y, w, h = box
        box_area = w * h
        aspect_ratio = w / h if h != 0 else 0
        
        if (min_ratio * image_area < box_area < max_ratio * image_area and
            0.1 < aspect_ratio < 10):  # Relaxed aspect ratio constraints
            filtered_boxes.append(box)
    
    # Merge nearby small components
    merged_boxes = merge_nearby_boxes(filtered_boxes, merge_distance)
    
    # Sort the merged boxes
    sorted_boxes = sort_bounding_boxes(merged_boxes)
    
    return sorted_boxes

def merge_nearby_boxes(boxes, distance):
    merged = []
    while boxes:
        base = boxes.pop(0)
        base_x, base_y, base_w, base_h = base
        
        to_merge = [base]
        i = 0
        while i < len(boxes):
            x, y, w, h = boxes[i]
            if (abs(x - base_x) < distance and abs(y - base_y) < distance) or \
               (abs(x + w - (base_x + base_w)) < distance and abs(y - base_y) < distance):
                to_merge.append(boxes.pop(i))
            else:
                i += 1
        
        if len(to_merge) > 1:
            x = min(box[0] for box in to_merge)
            y = min(box[1] for box in to_merge)
            max_x = max(box[0] + box[2] for box in to_merge)
            max_y = max(box[1] + box[3] for box in to_merge)
            merged.append((x, y, max_x - x, max_y - y))
        else:
            merged.append(base)
    
    return merged


def classify_symbol(image, box, padding=2):
    x, y, w, h = box
    symbol = image[y-padding:y+h+padding, x-padding:x+w+padding]
    
    if symbol.size == 0 or w == 0 or h == 0:
        return None

    aspect_ratio = w / h
    pixel_density = np.sum(symbol > 0) / (w * h)

    edges = cv2.Canny(symbol, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=20, minLineLength=w*0.5, maxLineGap=5)
    
    horizontal_lines = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < h * 0.2:
                horizontal_lines += 1

    if horizontal_lines >= 2 and aspect_ratio > 1.5:
        return "eq"
    elif horizontal_lines == 1 and 0.8 < aspect_ratio < 1.5 and pixel_density < 0.3:
        return "sub"
    else:
        return None



# Bounding box finding image with Connected Component Analysis (CCA)
def find_bb_cca(image_np, min_ratio=0.001, max_ratio=0.9):
    debug_dir = 'debug_images'

    # Apply adaptive thresholding
    binary = cv2.adaptiveThreshold(image_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    
    # Perform connected component analysis
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    
    # Filter components
    image_area = image_np.shape[0] * image_np.shape[1]
    bounding_boxes = []
    for i in range(1, num_labels):  # Skip the background label
        x, y, w, h, area = stats[i]
        if min_ratio * image_area < area < max_ratio * image_area:
            bounding_boxes.append((x, y, w, h))
    filtered_boxes = sort_bounding_boxes(bounding_boxes)

    image_with_boxes = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
    for box in filtered_boxes:
        x, y, w, h = box
        cv2.rectangle(image_with_boxes, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.imwrite(os.path.join(debug_dir, '7_image_with_contour_boxes.png'), image_with_boxes)


    return filtered_boxes


def pad_and_resize_element(element, target_size=(64, 64)): # pad element to square
    h, w = element.shape[:2]
    size = max(h, w)
    t = (size - h) // 2
    b = size - h - t
    l = (size - w) // 2
    r = size - w - l
    padded = cv2.copyMakeBorder(element, t, b, l, r, cv2.BORDER_CONSTANT, value=0)
    resized = cv2.resize(padded, target_size, interpolation=cv2.INTER_AREA)
    return resized


def preprocess_element(image, bbox, target_size=(64, 64)):
    x, y, w, h = bbox
    element = image[y:y+h, x:x+w]
    # Pad and resize the element
    element = pad_and_resize_element(element, target_size)
    # Reshape to (1, 64, 64) to match the expected input shape
    element = element.reshape((1, 64, 64)).astype(np.float32)
    element /= 255.0
    # Convert to PyTorch tensor and add batch dimension
    element_tensor = torch.from_numpy(element).float().unsqueeze(0)
    
    #visualize_preprocessed_element(element_tensor=element_tensor)

    return element_tensor


def predict_elements(image, bounding_boxes, model):
    results = []
    device = next(model.parameters()).device  # Get the device of the model
    
    for i, bbox in enumerate(bounding_boxes):
        element_tensor = preprocess_element(image, bbox).to(device)
        
        with torch.no_grad():
            output = model(element_tensor)
            probabilities = F.softmax(output, dim=1)
            predicted_class = probabilities.argmax(1).item()
            confidence = probabilities[0][predicted_class].item()
            
        results.append((predicted_class, confidence))
    return results


def load_model():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(script_dir, "normalized_model.pth")
    model = CNN(num_classes=19)  # Adjust num_classes if needed
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def visualize_preprocessed_image(image, bounding_boxes):
    plt.figure(figsize=(10, 10))
    plt.imshow(image, cmap='gray')
    
    # Draw bounding boxes
    for (x, y, w, h) in bounding_boxes:
        rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='red', linewidth=2)
        plt.gca().add_patch(rect)
    
    plt.title('Preprocessed Image with Bounding Boxes')
    plt.axis('off')
    plt.show()


def sort_bounding_boxes(bounding_boxes, x_threshold=20):
    def sort_key(box):
        x, y, w, h = box
        return x  # Sort primarily by x-coordinate

    return sorted(bounding_boxes, key=sort_key)



def predictions_to_string(predictions, bounding_boxes, symbols):
    result = []
    for (pred, _), bbox in zip(predictions, bounding_boxes):
        symbol = symbols[pred]
        if symbol in ['mul', '*', 'x']:
            result.append(('*', bbox))
        elif symbol in ['div', '/']:
            result.append(('/', bbox))
        elif symbol == 'sub':
            result.append(('-', bbox))
        else:
            result.append((symbol, bbox))
    return result


def detect_equals_sign(expression_with_bbox):
    new_expression = []
    i = 0
    while i < len(expression_with_bbox):
        if i + 1 < len(expression_with_bbox) and expression_with_bbox[i][0] == '-' and expression_with_bbox[i+1][0] == '-':
            bbox1 = expression_with_bbox[i][1]
            bbox2 = expression_with_bbox[i+1][1]
            
            # Check if the bounding boxes overlap horizontally
            if (bbox1[0] <= bbox2[0] + bbox2[2] and bbox2[0] <= bbox1[0] + bbox1[2]):
                # Combine the two subtraction symbols into an equals sign
                new_bbox = (
                    min(bbox1[0], bbox2[0]),
                    min(bbox1[1], bbox2[1]),
                    max(bbox1[0] + bbox1[2], bbox2[0] + bbox2[2]) - min(bbox1[0], bbox2[0]),
                    max(bbox1[1] + bbox1[3], bbox2[1] + bbox2[3]) - min(bbox1[1], bbox2[1])
                )
                new_expression.append(('=', new_bbox))
                i += 2
            else:
                new_expression.append(expression_with_bbox[i])
                i += 1
        else:
            new_expression.append(expression_with_bbox[i])
            i += 1
    return new_expression

def parse_expression(expression_string):
    # Replace 'mul' with '*', 'div' with '/', 'add' with '+', 'sub' with '-'
    expression_string = (expression_string
        .replace('mul', '*')
        .replace('div', '/')
        .replace('add', '+')
        .replace('sub', '-')
        .replace('eq', '=')
        .replace('dec', '.'))
    
    # Handle implicit multiplication (e.g., '2x' -> '2*x')
    expression_string = re.sub(r'(\d)([xyz])', r'\1*\2', expression_string)
    
    return expression_string

def post_process_expression(expression):
    # Replace consecutive subtraction signs with equals sign
    expression = expression.replace("--", "=")
    return expression



def evaluate_expression(parsed_expression):
    try:
        if not parsed_expression.strip():  # Check if the expression is empty or just whitespace
            return "Empty expression"
        
        # Define x, y, z as variables
        x, y, z = symbols('x y z')
        
        # Parse the expression
        expr = parse_expr(parsed_expression)
        
        # If the expression contains variables, return it as a string
        if expr.free_symbols:
            return str(expr)
        
        # Evaluate the expression
        result = expr.evalf()
        
        # Convert to int if it's a whole number
        if result.is_integer:
            return int(result)
        
        # Otherwise, round to 4 decimal places
        return round(float(result), 4)
    
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"


if __name__ == "__main__":
    print("no CLI running support right now")