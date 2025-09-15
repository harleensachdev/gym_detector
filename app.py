from flask import Flask, request, jsonify
import os
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
import sys
import time
import subprocess
import json
from io import BytesIO
from PIL import Image
import logging
import requests
from pathlib import Path
import numpy as np

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

# Use your ONNX model from Google Drive
# Get the direct download link from your Google Drive ONNX model
MODEL_URL = os.getenv("MODEL_URL", "https://drive.google.com/uc?export=download&id=1JBZHN-gwW4ukfMbfI7N-gIlU8iaXp1I_")
CLASS_NAMES = [
    "arm_curl",
    "chest_fly",
    "chest_press",
    "chin_dip",
    "dumbell",
    "lat_pulldown",
    "lateral_raises",
    "leg_curl",
    "leg_extension",
    "leg_press",
    "seated_cable_row",
    "seated_dip",
    "shoulder_press",
    "smith_machine"
  ]  # Update with your actual classes

app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

def install_minimal_dependencies():
    """Install only minimal dependencies for inference"""
    global DEPENDENCIES_INSTALLED, ML_ERROR
    
    if DEPENDENCIES_INSTALLED:
        return True
    
    logger.info("Installing minimal ML dependencies...")
    
    try:
        # Only install what we absolutely need
        packages = [
            'numpy==1.24.3',
            'onnxruntime==1.16.0',  # Much smaller than PyTorch
            'scipy==1.11.4',        # For NMS if needed
        ]
        
        logger.info("Installing minimal packages...")
        result = subprocess.run(
            [sys.executable, '-m', 'pip', 'install', '--no-cache-dir'] + packages,
            capture_output=True,
            text=True,
            timeout=120
        )
        
        if result.returncode != 0:
            logger.error(f"Installation failed: {result.stderr}")
            ML_ERROR = "Failed to install minimal ML packages"
            return False
            
        logger.info("✅ Minimal dependencies installed successfully!")
        DEPENDENCIES_INSTALLED = True
        return True
            
    except subprocess.TimeoutExpired:
        logger.error("Installation timed out")
        ML_ERROR = "Dependency installation timed out"
        return False
    except Exception as e:
        logger.error(f"Installation error: {e}")
        ML_ERROR = str(e)
        return False

def load_onnx_model():
    """Load lightweight ONNX model"""
    global ML_READY, ML_LOADING, ML_ERROR, onnx_session
    
    if ML_READY or ML_LOADING:
        return ML_READY
    
    ML_LOADING = True
    
    try:
        # Install minimal dependencies
        if not install_minimal_dependencies():
            ML_LOADING = False
            return False
        
        import onnxruntime as ort
        
        # Download ONNX model (much smaller than PyTorch)
        model_path = "/tmp/model.onnx"
        if not os.path.exists(model_path):
            logger.info("Downloading ONNX model...")
            response = requests.get(MODEL_URL, timeout=60, stream=True)
            response.raise_for_status()
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            logger.info(f"ONNX model downloaded: {os.path.getsize(model_path) / 1024 / 1024:.1f}MB")
        
        # Load ONNX model
        onnx_session = ort.InferenceSession(model_path)
        logger.info("✅ ONNX Model loaded successfully!")
        
        ML_READY = True
        ML_LOADING = False
        return True
        
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        ML_ERROR = str(e)
        ML_LOADING = False
        return False

def preprocess_image(image):
    """Minimal image preprocessing for YOLO"""
    # Resize image
    if max(image.size) > MAX_IMAGE_SIZE:
        ratio = MAX_IMAGE_SIZE / max(image.size)
        new_size = tuple(int(dim * ratio) for dim in image.size)
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    # Convert to RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert to numpy array and normalize
    img_array = np.array(image).astype(np.float32)
    img_array = img_array / 255.0  # Normalize to 0-1
    
    # Add batch dimension and transpose to CHW format
    img_array = img_array.transpose(2, 0, 1)  # HWC to CHW
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    
    return img_array, image.size

def postprocess_detections(outputs, original_size, conf_threshold=0.25):
    """Process ONNX model outputs"""
    # This depends on your specific ONNX model output format
    # Typical YOLO output is [batch, detections, 5+classes]
    detections = []
    
    try:
        predictions = outputs[0][0]  # Remove batch dimension
        
        for detection in predictions:
            confidence = detection[4]  # Objectness score
            
            if confidence > conf_threshold:
                # Extract bounding box (assuming normalized coordinates)
                x_center, y_center, width, height = detection[:4]
                
                # Get class scores
                class_scores = detection[5:]
                class_id = np.argmax(class_scores)
                class_confidence = class_scores[class_id]
                
                final_confidence = confidence * class_confidence
                
                if final_confidence > conf_threshold:
                    detections.append({
                        "className": CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}",
                        "confidence": float(final_confidence),
                        "boundingBox": {
                            "x": float(x_center - width/2),
                            "y": float(y_center - height/2),
                            "width": float(width),
                            "height": float(height)
                        }
                    })
    
    except Exception as e:
        logger.error(f"Postprocessing error: {e}")
    
    return detections

# Basic Flask routes
@app.route('/', methods=['GET'])
def health():
    """Health check"""
    uptime = time.time() - start_time
    
    return jsonify({
        "status": "healthy",
        "ml_ready": ML_READY,
        "ml_loading": ML_LOADING,
        "ml_error": ML_ERROR,
        "uptime": round(uptime, 1),
        "message": "Lightweight ONNX API is running",
        "model_type": "ONNX (minimal dependencies)",
        "endpoints": ["/", "/detect"]
    })

@app.route('/detect', methods=['POST'])
def detect():
    """Lightweight ML detection using ONNX"""
    
    # Load ONNX model on first request
    if not ML_READY:
        if ML_LOADING:
            return jsonify({
                "success": False,
                "error": "Model is loading, please wait...",
                "retry_after": 30,
                "status": "loading"
            }), 503
        
        if not load_onnx_model():
            return jsonify({
                "success": False,
                "error": "Model initialization failed",
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
        
        # Preprocess for ONNX model
        processed_image, original_size = preprocess_image(image)
        
        # Run ONNX inference
        input_name = onnx_session.get_inputs()[0].name
        outputs = onnx_session.run(None, {input_name: processed_image})
        
        # Postprocess results
        detections = postprocess_detections(outputs, original_size)
        
        return jsonify({
            "success": True,
            "detections": detections,
            "count": len(detections),
            "image_size": list(original_size),
            "model_type": "ONNX"
        })
        
    except NameError:
        return jsonify({
            "success": False,
            "error": "ONNX model not ready",
            "status": "not_ready"
        }), 503
    except Exception as e:
        logger.error(f"Detection error: {e}")
        return jsonify({"error": f"Detection failed: {str(e)}"}), 500

# Error handlers
@app.errorhandler(413)
def file_too_large(e):
    return jsonify({"error": "File too large (max 5MB)"}), 413

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal error: {e}")
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    logger.info("🚀 Starting Ultra-Lightweight ONNX API Server...")
    logger.info("Dependencies: numpy, onnxruntime, pillow only")
    
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"Starting server on port {port}")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        threaded=True
    )