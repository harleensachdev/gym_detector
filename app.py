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
import numpy as np

# Configure logging for Railway
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables
ML_READY = False
ML_LOADING = False
ML_ERROR = None
start_time = time.time()
onnx_session = None

# Railway-optimized configuration
MAX_IMAGE_SIZE = 640
MAX_CONTENT_LENGTH = 2 * 1024 * 1024  # 2MB limit for Railway

# Use your ONNX model from Google Drive
MODEL_URL = os.getenv("MODEL_URL", "https://drive.google.com/uc?export=download&id=1JBZHN-gwW4ukfMbfI7N-gIlU8iaXp1I_")
CLASS_NAMES = [
    "arm_curl", "chest_fly", "chest_press", "chin_dip", "dumbell",
    "lat_pulldown", "lateral_raises", "leg_curl", "leg_extension",
    "leg_press", "seated_cable_row", "seated_dip", "shoulder_press", "smith_machine"
]

app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

def load_onnx_model():
    """Load lightweight ONNX model with error handling"""
    global ML_READY, ML_LOADING, ML_ERROR, onnx_session
    
    if ML_READY or ML_LOADING:
        return ML_READY
    
    ML_LOADING = True
    logger.info("🔄 Loading ONNX model...")
    
    try:
        import onnxruntime as ort
        
        # Download ONNX model if not exists
        model_path = "/tmp/model.onnx"
        if not os.path.exists(model_path):
            logger.info("📥 Downloading 5MB ONNX model...")
            
            # Add timeout and retry logic
            for attempt in range(3):
                try:
                    response = requests.get(MODEL_URL, timeout=60, stream=True)
                    response.raise_for_status()
                    
                    with open(model_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    
                    model_size = os.path.getsize(model_path) / 1024 / 1024
                    logger.info(f"✅ Model downloaded: {model_size:.1f}MB")
                    break
                    
                except Exception as e:
                    logger.warning(f"Download attempt {attempt + 1} failed: {e}")
                    if attempt == 2:
                        raise
                    time.sleep(2)
        
        # Load ONNX model with optimizations
        providers = ['CPUExecutionProvider']
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        onnx_session = ort.InferenceSession(
            model_path, 
            sess_options=sess_options,
            providers=providers
        )
        
        logger.info("✅ ONNX Model loaded successfully!")
        ML_READY = True
        ML_LOADING = False
        return True
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        ML_ERROR = str(e)
        ML_LOADING = False
        return False

def preprocess_image(image):
    """Optimized image preprocessing for Railway"""
    # Resize to save memory
    if max(image.size) > MAX_IMAGE_SIZE:
        ratio = MAX_IMAGE_SIZE / max(image.size)
        new_size = tuple(int(dim * ratio) for dim in image.size)
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Efficient numpy conversion
    img_array = np.array(image, dtype=np.float32) / 255.0
    
    # YOLO format: CHW with batch dimension
    img_array = img_array.transpose(2, 0, 1)[np.newaxis, :]
    
    return img_array, image.size

def postprocess_detections(outputs, original_size, conf_threshold=0.25, nms_threshold=0.4):
    """Process ONNX model outputs with NMS"""
    detections = []
    
    try:
        predictions = outputs[0][0]  # Remove batch dimension
        
        # Collect valid detections
        valid_detections = []
        for detection in predictions:
            confidence = detection[4]  # Objectness score
            
            if confidence > conf_threshold:
                # Extract bounding box
                x_center, y_center, width, height = detection[:4]
                
                # Get best class
                class_scores = detection[5:]
                class_id = np.argmax(class_scores)
                class_confidence = class_scores[class_id]
                
                final_confidence = confidence * class_confidence
                
                if final_confidence > conf_threshold:
                    valid_detections.append({
                        "className": CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}",
                        "confidence": float(final_confidence),
                        "boundingBox": {
                            "x": float(max(0, x_center - width/2)),
                            "y": float(max(0, y_center - height/2)),
                            "width": float(min(1, width)),
                            "height": float(min(1, height))
                        }
                    })
        
        # Simple confidence-based filtering instead of full NMS to save computation
        valid_detections.sort(key=lambda x: x["confidence"], reverse=True)
        detections = valid_detections[:10]  # Limit to top 10 detections
    
    except Exception as e:
        logger.error(f"❌ Postprocessing error: {e}")
    
    return detections

# Flask routes
@app.route('/', methods=['GET'])
def health():
    """Railway-compatible health check"""
    uptime = time.time() - start_time
    
    return jsonify({
        "status": "healthy",
        "service": "Gym Equipment Detection API",
        "ml_ready": ML_READY,
        "ml_loading": ML_LOADING,
        "ml_error": ML_ERROR,
        "uptime_seconds": round(uptime, 1),
        "model_type": "ONNX (5MB)",
        "platform": "Railway",
        "endpoints": {
            "health": "/",
            "detect": "/detect"
        },
        "limits": {
            "max_image_size": f"{MAX_CONTENT_LENGTH // 1024 // 1024}MB",
            "max_resolution": f"{MAX_IMAGE_SIZE}px"
        }
    })

@app.route('/detect', methods=['POST'])
def detect():
    """Lightweight ML detection endpoint"""
    
    # Auto-load model on first request
    if not ML_READY and not ML_LOADING:
        logger.info("🚀 First request - initializing model...")
        if not load_onnx_model():
            return jsonify({
                "success": False,
                "error": "Model initialization failed",
                "details": ML_ERROR,
                "retry_after": 60
            }), 500
    
    # Handle loading state
    if ML_LOADING:
        return jsonify({
            "success": False,
            "error": "Model is still loading, please wait...",
            "retry_after": 30,
            "status": "loading"
        }), 503
    
    # Handle failed state
    if not ML_READY:
        return jsonify({
            "success": False,
            "error": "Model not available",
            "details": ML_ERROR
        }), 503
    
    try:
        # Get image data
        if 'image' in request.files:
            image_data = request.files['image'].read()
        elif request.content_type and 'image' in request.content_type:
            image_data = request.get_data()
        else:
            return jsonify({
                "success": False,
                "error": "No image provided. Send as 'image' file or in request body."
            }), 400
        
        if not image_data:
            return jsonify({"success": False, "error": "Empty image data"}), 400
        
        # Process image
        try:
            image = Image.open(BytesIO(image_data))
            logger.info(f"📸 Processing image: {image.size}")
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Invalid image format: {str(e)}"
            }), 400
        
        # Preprocess for model
        processed_image, original_size = preprocess_image(image)
        
        # Run inference
        input_name = onnx_session.get_inputs()[0].name
        start_inference = time.time()
        outputs = onnx_session.run(None, {input_name: processed_image})
        inference_time = time.time() - start_inference
        
        # Process results
        detections = postprocess_detections(outputs, original_size)
        
        logger.info(f"✅ Detection complete: {len(detections)} objects found in {inference_time:.3f}s")
        
        return jsonify({
            "success": True,
            "detections": detections,
            "count": len(detections),
            "metadata": {
                "image_size": list(original_size),
                "inference_time_ms": round(inference_time * 1000, 2),
                "model_type": "ONNX",
                "platform": "Railway"
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Detection error: {e}")
        return jsonify({
            "success": False,
            "error": f"Detection failed: {str(e)}"
        }), 500

# Error handlers
@app.errorhandler(413)
def file_too_large(e):
    return jsonify({
        "success": False,
        "error": f"File too large. Maximum size: {MAX_CONTENT_LENGTH // 1024 // 1024}MB"
    }), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "success": False,
        "error": "Endpoint not found",
        "available_endpoints": ["/", "/detect"]
    }), 404

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"❌ Internal error: {e}")
    return jsonify({
        "success": False,
        "error": "Internal server error"
    }), 500

# Auto-load model on startup for Railway
def initialize_model():
    """Initialize model on startup"""
    logger.info("🚀 Railway startup - pre-loading model...")
    load_onnx_model()

if __name__ == '__main__':
    logger.info("🚀 Starting Ultra-Lightweight Gym Detection API")
    logger.info("📦 Dependencies: Flask, ONNX Runtime, PIL, NumPy only")
    logger.info("🏋️ Model: 5MB Gym Equipment Detection")
    
    # Get port from Railway environment
    port = int(os.environ.get('PORT', 5000))
    
    # Pre-load model for Railway
    initialize_model()
    
    logger.info(f"🌐 Server starting on port {port}")
    
    # Railway-optimized settings
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        threaded=True,
        use_reloader=False
    )