const canvas = document.getElementById('sketchpad');
const ctx = canvas.getContext('2d');
const evaluateBtn = document.getElementById('evaluateBtn');
const clearBtn = document.getElementById('clearBtn');
const showPreprocessingBtn = document.getElementById('showPreprocessingBtn');
const resultP = document.getElementById('result');
const preprocessingModal = document.getElementById('preprocessingModal');
const preprocessingContent = document.getElementById('preprocessingContent');
const closeModalBtn = document.getElementById('closeModalBtn');

let isDrawing = false;
let lastX = 0;
let lastY = 0;

function resizeCanvas() {
    const container = canvas.parentElement;
    canvas.width = container.clientWidth;
    canvas.height = container.clientWidth / 2; // 2:1 aspect ratio
    clearCanvas();
}

// Call resizeCanvas initially and add event listener for window resize
resizeCanvas();
window.addEventListener('resize', resizeCanvas);

// Set up event listeners
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);
evaluateBtn.addEventListener('click', evaluateExpression);
clearBtn.addEventListener('click', clearCanvas);
showPreprocessingBtn.addEventListener('click', showPreprocessing);
closeModalBtn.addEventListener('click', closePreprocessingModal);

function getMousePos(canvas, evt) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return {
        x: (evt.clientX - rect.left) * scaleX,
        y: (evt.clientY - rect.top) * scaleY
    };
}

function startDrawing(e) {
    isDrawing = true;
    const pos = getMousePos(canvas, e);
    [lastX, lastY] = [pos.x, pos.y];
    // Draw a single point in case of a click without drag
    ctx.beginPath();
    ctx.arc(lastX, lastY, ctx.lineWidth / 2, 0, Math.PI * 2);
    ctx.fill();
}

function draw(e) {
    if (!isDrawing) return;
    
    const pos = getMousePos(canvas, e);
    const x = pos.x;
    const y = pos.y;

    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(x, y);
    ctx.stroke();

    [lastX, lastY] = [x, y];
}

function stopDrawing() {
    isDrawing = false;
    // Reset lastX and lastY to prevent connecting to the last point
    lastX = 0;
    lastY = 0;
}

function clearCanvas() {
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    resultP.textContent = '';
}

async function evaluateExpression() {
    const imageData = canvas.toDataURL('image/png');
    
    if (isCanvasBlank()) {
        resultP.textContent = 'Please draw something before evaluating.';
        return;
    }
    
    try {
        const response = await fetch('http://127.0.0.1:5000/evaluate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageData }),
            credentials: 'include',
            mode: 'cors'
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        resultP.innerHTML = `Expression: ${result.expression}<br>Result: ${result.result}`;
        
        // Store the preprocessing data
        window.preprocessingData = result.preprocessing;
    } catch (error) {
        console.error('Error:', error);
        resultP.textContent = 'An error occurred during evaluation: ' + error.message;
    }
}

function showPreprocessing() {
    if (!window.preprocessingData) {
        alert('No preprocessing data available. Please evaluate an expression first.');
        return;
    }
    
    preprocessingContent.innerHTML = '';
    
    // Display the processed image with bounding boxes
    const processedImg = document.createElement('img');
    processedImg.src = window.preprocessingData.processed_image;
    processedImg.className = 'preprocessed-image';
    preprocessingContent.appendChild(processedImg);
    
    // Display individual predictions
    const predictionsContainer = document.createElement('div');
    predictionsContainer.className = 'predictions-grid';
    
    window.preprocessingData.predictions.forEach((prediction) => {
        const predictionElement = document.createElement('div');
        predictionElement.className = 'prediction-item';
        predictionElement.innerHTML = `
            <img src="${prediction.image}" class="prediction-image" alt="Predicted symbol" />
            <div class="prediction-text">
                <p><strong>Predicted:</strong> ${prediction.symbol}</p>
                <p><strong>Confidence:</strong> ${prediction.confidence.toFixed(2)}</p>
            </div>
        `;
        predictionsContainer.appendChild(predictionElement);
    });
    
    preprocessingContent.appendChild(predictionsContainer);
    
    preprocessingModal.style.display = 'flex';
}


function closePreprocessingModal() {
    preprocessingModal.style.display = 'none';
}

function isCanvasBlank() {
    const blankCanvas = document.createElement('canvas');
    blankCanvas.width = canvas.width;
    blankCanvas.height = canvas.height;
    return canvas.toDataURL() === blankCanvas.toDataURL();
}

// Set initial canvas style
ctx.lineWidth = 7;
ctx.lineCap = 'round';
ctx.strokeStyle = 'black';

clearCanvas();