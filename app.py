from flask import Flask, request, jsonify
import os
import sys
import time
import subprocess
import json
from io import BytesIO
from PIL import Image
import logging
import requests
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables
ML_READY = False
ML_LOADING = False
ML_ERROR = None
start_time = time.time()
DEPENDENCIES_INSTALLED = False

# Configuration
MAX_IMAGE_SIZE = 640
MAX_CONTENT_LENGTH = 5 * 1024 * 1024
MODEL_URL = os.getenv("MODEL_URL", "https://drive.google.com/uc?export=download&id=1A9z88pRkkWwdF0LNUMxlVq11kG8nNI60")

app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

def install_ml_dependencies():
    """Install ML dependencies at runtime to avoid build timeout"""
    global DEPENDENCIES_INSTALLED, ML_ERROR
    
    if DEPENDENCIES_INSTALLED:
        return True
    
    logger.info("Installing ML dependencies at runtime...")
    
    try:
        # Install minimal packages first
        packages = [
            'numpy==1.24.3',
            'ultralytics==8.0.200',
            'opencv-python-headless==4.8.1.78',
        ]
        
        # Install torch separately with CPU-only version
        torch_packages = [
            '--index-url', 'https://download.pytorch.org/whl/cpu',
            'torch==2.0.1+cpu',
            'torchvision==0.15.2+cpu'
        ]
        
        logger.info("Installing basic packages...")
        result1 = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '--no-cache-dir'] + packages,
            capture_output=True,
            text=True,
            timeout=180  # 3 minutes
        )
        
        if result1.returncode != 0:
            logger.error(f"Basic packages failed: {result1.stderr}")
            ML_ERROR = "Failed to install basic ML packages"
            return False
        
        logger.info("Installing PyTorch CPU...")
        result2 = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '--no-cache-dir'] + torch_packages,
            capture_output=True,
            text=True,
            timeout=180  # 3 minutes
        )
        
        if result2.returncode != 0:
            logger.error(f"PyTorch installation failed: {result2.stderr}")
            ML_ERROR = "Failed to install PyTorch"
            return False
            
        logger.info("✅ ML dependencies installed successfully!")
        DEPENDENCIES_INSTALLED = True
        return True
            
    except subprocess.TimeoutExpired:
        logger.error("Installation timed out")
        ML_ERROR = "ML dependency installation timed out"
        return False
    except Exception as e:
        logger.error(f"Installation error: {e}")
        ML_ERROR = str(e)
        return False

def load_model_lazy():
    """Load model only when needed"""
    global ML_READY, ML_LOADING, ML_ERROR
    
    if ML_READY or ML_LOADING:
        return ML_READY
    
    ML_LOADING = True
    
    try:
        # First install dependencies if needed
        if not install_ml_dependencies():
            ML_LOADING = False
            return False
        
        # Now import and load model
        from ultralytics import YOLO
        
        # Download model if needed
        model_path = "/tmp/model.pt"
        if not os.path.exists(model_path):
            logger.info("Downloading model...")
            response = requests.get(MODEL_URL, timeout=120, stream=True)
            response.raise_for_status()
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"Model downloaded: {os.path.getsize(model_path) / 1024 / 1024:.1f}MB")
        
        # Load model
        global model
        model = YOLO(model_path)
        logger.info("✅ Model loaded successfully!")
        
        ML_READY = True
        ML_LOADING = False
        return True
        
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        ML_ERROR = str(e)
        ML_LOADING = False
        return False

# Basic Flask routes that work without ML
@app.route('/', methods=['GET'])
def health():
    """Health check - always works"""
    uptime = time.time() - start_time
    
    return jsonify({
        "status": "healthy",
        "ml_ready": ML_READY,
        "ml_loading": ML_LOADING,
        "ml_error": ML_ERROR,
        "uptime": round(uptime, 1),
        "message": "API is running. ML features will load on first use.",
        "endpoints": ["/", "/quick-test", "/process-image", "/detect", "/install-ml"]
    })

@app.route('/quick-test', methods=['GET'])
def quick_test():
    """Quick test that doesn't need ML"""
    return jsonify({
        "success": True,
        "message": "Server is responsive!",
        "ml_available": ML_READY,
        "timestamp": time.time()
    })

@app.route('/process-image', methods=['POST'])
def process_image():
    """Simple image processing without ML"""
    try:
        # Get image
        if 'image' in request.files:
            file = request.files['image']
            image_data = file.read()
        elif request.content_type and 'image' in request.content_type:
            image_data = request.get_data()
        else:
            return jsonify({"error": "No image data found"}), 400
        
        if not image_data:
            return jsonify({"error": "Empty image data"}), 400
        
        # Basic image processing
        try:
            image = Image.open(BytesIO(image_data))
        except Exception as e:
            return jsonify({"error": f"Invalid image format: {e}"}), 400
        
        # Get image info
        width, height = image.size
        mode = image.mode
        
        # Resize if needed
        if max(width, height) > MAX_IMAGE_SIZE:
            ratio = MAX_IMAGE_SIZE / max(width, height)
            new_size = (int(width * ratio), int(height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        return jsonify({
            "success": True,
            "original_size": [width, height],
            "processed_size": list(image.size),
            "mode": mode,
            "format": image.format,
            "message": "Image processed successfully"
        })
        
    except Exception as e:
        logger.error(f"Image processing error: {e}")
        return jsonify({"error": f"Processing failed: {str(e)}"}), 500

@app.route('/detect', methods=['POST'])
def detect():
    """ML detection - loads dependencies on first use"""
    
    # Load ML on first request
    if not ML_READY:
        if ML_LOADING:
            return jsonify({
                "success": False,
                "error": "ML is loading, please wait and retry in 60 seconds...",
                "retry_after": 60,
                "status": "loading"
            }), 503
        
        # Start loading
        logger.info("First ML request - installing dependencies...")
        
        # Try to load ML (this will take time)
        if not load_model_lazy():
            return jsonify({
                "success": False,
                "error": "ML initialization failed. The server is still running but ML features are unavailable.",
                "details": ML_ERROR,
                "status": "failed"
            }), 500
    
    try:
        # Get image
        if 'image' in request.files:
            file = request.files['image']
            image_data = file.read()
        elif request.content_type and 'image' in request.content_type:
            image_data = request.get_data()
        else:
            return jsonify({"error": "No image data found"}), 400
        
        if not image_data:
            return jsonify({"error": "Empty image data"}), 400
        
        # Process image
        try:
            image = Image.open(BytesIO(image_data))
        except Exception as e:
            return jsonify({"error": f"Invalid image format: {e}"}), 400
        
        # Resize if needed
        if max(image.size) > MAX_IMAGE_SIZE:
            ratio = MAX_IMAGE_SIZE / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Run detection
        results = model(image, conf=0.25, device='cpu', verbose=False)
        
        # Parse results
        detections = []
        for r in results:
            if r.boxes is not None:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxyn[0].tolist()
                    detections.append({
                        "className": model.names[int(box.cls[0])],
                        "confidence": float(box.conf[0]),
                        "boundingBox": {
                            "x": (x1 + x2) / 2 - (x2 - x1) / 2,
                            "y": (y1 + y2) / 2 - (y2 - y1) / 2,
                            "width": x2 - x1,
                            "height": y2 - y1
                        }
                    })
        
        return jsonify({
            "success": True,
            "detections": detections,
            "count": len(detections),
            "image_size": list(image.size)
        })
        
    except NameError:
        # Model not loaded yet
        return jsonify({
            "success": False,
            "error": "ML model not ready yet. Please try again in a few seconds.",
            "status": "not_ready"
        }), 503
    except Exception as e:
        logger.error(f"Detection error: {e}")
        return jsonify({"error": f"Detection failed: {str(e)}"}), 500

@app.route('/install-ml', methods=['POST'])
def install_ml():
    """Manually trigger ML installation"""
    if ML_READY:
        return jsonify({"message": "ML already ready", "status": "ready"})
    
    if ML_LOADING:
        return jsonify({"message": "ML is loading...", "status": "loading"})
    
    # Start installation in background
    import threading
    thread = threading.Thread(target=load_model_lazy, daemon=True)
    thread.start()
    
    return jsonify({
        "message": "ML installation started. This will take 2-3 minutes.",
        "check_status": "GET /",
        "status": "started"
    })

# Error handlers
@app.errorhandler(413)
def file_too_large(e):
    return jsonify({"error": "File too large (max 5MB)"}), 413

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal error: {e}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    logger.info("🚀 Starting Ultra-Light API Server...")
    logger.info("Server will start immediately. ML features will be installed on first use.")
    
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting server on port {port}")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        threaded=True
    )