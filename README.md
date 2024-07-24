# Math Expression Evaluator

An interactive web application that evaluates handwritten mathematical expressions using machine learning and vanilla JavaScript.

## Description

This project combines a Flask backend with a vanilla JavaScript frontend to create a unique mathematical tool. Users can draw mathematical expressions on a canvas, which are then processed, recognized, and evaluated in real-time. The application uses a custom-trained Convolutional Neural Network (CNN) to recognize handwritten digits and mathematical symbols.

## Tech Stack

- Frontend: HTML, CSS, Vanilla JavaScript
- Backend: Python, Flask
- Machine Learning: PyTorch, OpenCV
- Additional Libraries: NumPy, Sympy

## Setup and Installation

1. Clone the repository:
    git clone https://github.com/yourusername/math-expression-evaluator.git
    cd math-expression-evaluator

2. Set up a Python virtual environment:
    python -m venv venv
    source venv/bin/activate  # On Windows use venv\Scripts\activate

3. Install the required Python packages:
    pip install -r requirements.txt
   
5. Start the Flask server:
    python app.py

6. Open `index.html` in your web browser or use a local server to serve the frontend files.

## Features

- Real-time handwriting recognition
- Mathematical expression parsing and evaluation
- Detailed preprocessing visualization
- Responsive canvas for drawing expressions

## Demo

![Math Expression Evaluator Demo](demo.gif)

*GIF showcasing the application in use*

## Challenges and Solutions

The most challenging aspect of this project was implementing accurate segmentation and recognition of handwritten mathematical symbols. To overcome this, we developed a custom image processing pipeline using OpenCV for segmentation and a CNN trained on a dataset of handwritten mathematical symbols. 

Another significant challenge was creating a smooth drawing experience and integrating the machine learning model with the web interface in real-time. We solved this by optimizing the backend processing and using asynchronous JavaScript to handle the communication between the frontend and backend efficiently, all without relying on any frontend frameworks.

## Future Improvements

- Expand the range of recognized mathematical symbols and operations
- Implement user accounts to save and share expressions
- Add support for more complex mathematical operations and functions
- Optimize the model for better performance on mobile devices
- Enhance the drawing experience with advanced canvas features

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to check [issues page](https://github.com/yourusername/math-expression-evaluator/issues) if you want to contribute.

## License

This project is [MIT](https://choosealicense.com/licenses/mit/) licensed.
