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

async function evaluateExpression() {
    const imageData = canvas.toDataURL('image/png');
    
    try {
        const response = await fetch('http://localhost:5000/evaluate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ image: imageData }),
        });
        
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        
        const result = await response.json();
        resultP.textContent = `Expression: ${result.expression}, Result: ${result.result}`;
    } catch (error) {
        console.error('Error:', error);
        resultP.textContent = 'An error occurred during evaluation.';
    }
}

// Set initial canvas style
ctx.lineWidth = 10;
ctx.lineCap = 'round';
ctx.strokeStyle = 'black';