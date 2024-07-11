const canvas = document.getElementById('sketchpad');
const ctx = canvas.getContext('2d');
const evaluateBtn = document.getElementById('evaluateBtn');
const clearBtn = document.getElementById('clearBtn');
const resultP = document.getElementById('result');

let isDrawing = false;

// Set up event listeners
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);
evaluateBtn.addEventListener('click', evaluateExpression);
clearBtn.addEventListener('click', clearCanvas);

function startDrawing(e) {
    isDrawing = true;
    draw(e);
}

function draw(e) {
    if (!isDrawing) return;

    ctx.lineWidth = 2;
    ctx.lineCap = 'round';
    ctx.strokeStyle = 'black';

    ctx.lineTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
}

function stopDrawing() {
    isDrawing = false;
    ctx.beginPath();
}

function clearCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    resultP.textContent = '';
}

function evaluateExpression() {
    // Here you would typically send the canvas image to your server for processing
    // For now, we'll just display a placeholder message
    resultP.textContent = 'Evaluation functionality not implemented yet.';
    
    // When your model is ready, you can implement the actual evaluation logic here
    // This might involve:
    // 1. Converting the canvas to an image (e.g., using canvas.toDataURL())
    // 2. Sending the image to your server (e.g., using fetch())
    // 3. Processing the image on the server with your ML model
    // 4. Sending the result back to the client
    // 5. Displaying the result in the resultP element
}