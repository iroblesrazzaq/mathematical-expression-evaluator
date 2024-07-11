const canvas = document.getElementById('sketchpad');
const ctx = canvas.getContext('2d');
const evaluateBtn = document.getElementById('evaluateBtn');
const clearBtn = document.getElementById('clearBtn');
const resultP = document.getElementById('result');

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

function getMousePos(canvas, evt) {
    const rect = canvas.getBoundingClientRect();
    return {
        x: evt.clientX - rect.left,
        y: evt.clientY - rect.top
    };
}

function startDrawing(e) {
    isDrawing = true;
    [lastX, lastY] = [e.offsetX, e.offsetY];
}

function draw(e) {
    if (!isDrawing) return;

    ctx.beginPath();
    ctx.moveTo(lastX, lastY);
    ctx.lineTo(e.offsetX, e.offsetY);
    ctx.stroke();

    [lastX, lastY] = [e.offsetX, e.offsetY];
}

function stopDrawing() {
    isDrawing = false;
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

// Set initial canvas style
ctx.lineWidth = 10;
ctx.lineCap = 'round';
ctx.strokeStyle = 'black';