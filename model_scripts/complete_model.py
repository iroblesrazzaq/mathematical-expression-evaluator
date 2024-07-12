
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

def process_image(image_data, target_size=200):
    # Decode the base64 image
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
    image_np = cv2.resize(image_np, (target_size, target_size), interpolation=cv2.INTER_AREA)
    # Save the resized image
    # cv2.imwrite('debug_images/resized_image.png', image_np)

    return image_np


# Bounding box finding image with Connected Component Analysis (CCA)
def find_bb_contour(image_np, min_ratio=0.001, max_ratio=0.9):
    binary = cv2.adaptiveThreshold(image_np, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    
    # Save the binary image
    #cv2.imwrite('debug_images/binary_image.png', binary)
    
    # Find contours and get bounding boxes
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
    
    # Filter bounding boxes
    image_area = image_np.shape[0] * image_np.shape[1]
    bounding_boxes = [box for box in bounding_boxes if 
                      min_ratio * image_area < box[2] * box[3] < max_ratio * image_area]
    bounding_boxes = sort_bounding_boxes(bounding_boxes)

    #visualize_preprocessed_image(image, bounding_boxes=bounding_boxes)
    return bounding_boxes


# Bounding box finding image with Connected Component Analysis (CCA)
def find_bb_cca(image_np, min_ratio=0.001, max_ratio=0.9):
    
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
    bounding_boxes = sort_bounding_boxes(bounding_boxes)
    return bounding_boxes


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


def predictions_to_string(predictions, symbols):
    if not predictions:
        return ""  # Return an empty string if there are no predictions
    result = []
    for pred, _ in predictions:
        symbol = symbols[pred]
        if symbol in ['mul', '*', 'x']:  # Handle multiplication symbols
            result.append('*')
        elif symbol in ['div', '/']:  # Handle division symbols
            result.append('/')
        else:
            result.append(symbol)
    return ''.join(result)


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
        
        # Otherwise, evaluate it
        return expr.evalf()
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"


if __name__ == "__main__":
    print("no CLI running support right now")