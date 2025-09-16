from flask import Flask, request, jsonify
import os
import sys
import time
import logging
from io import BytesIO
from PIL import Image
import requests
import numpy as np
import cv2

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
net = None

# Configuration
MAX_IMAGE_SIZE = 640
MAX_CONTENT_LENGTH = 2 * 1024 * 1024

# Your ONNX model URL
MODEL_URL = os.getenv("MODEL_URL", "https://drive.google.com/uc?export=download&id=10x_l-SAe3uuoGjIK_wZIOhQ0WBTx79i7")
CLASS_NAMES = [
    "arm_curl", "chest_fly", "chest_press", "chin_dip", "dumbell",
    "lat_pulldown", "lateral_raises", "leg_curl", "leg_extension",
    "leg_press", "seated_cable_row", "seated_dip", "shoulder_press", "smith_machine"
]

app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

def load_opencv_model():
    """Load ONNX model using OpenCV DNN - Railway compatible!"""
    global ML_READY, ML_LOADING, ML_ERROR, net
    
    if ML_READY or ML_LOADING:
        return ML_READY
    
    ML_LOADING = True
    logger.info("🚀 Loading 5MB ONNX model with OpenCV DNN (Railway compatible)...")
    
    try:
        # Download model
        model_path = "/tmp/model.onnx"
        if not os.path.exists(model_path):
            logger.info("📥 Downloading 5MB model from Google Drive...")
            start_download = time.time()
            
            response = requests.get(
                MODEL_URL, 
                timeout=30,
                stream=True,
                headers={'User-Agent': 'Railway-OpenCV-Server/1.0'}
            )
            response.raise_for_status()
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=32768):
                    if chunk:
                        f.write(chunk)
            
            download_time = time.time() - start_download
            model_size = os.path.getsize(model_path) / 1024 / 1024
            logger.info(f"✅ Model downloaded: {model_size:.1f}MB in {download_time:.1f}s")
        
        # Load with OpenCV DNN (no executable stack issues!)
        logger.info("🔄 Loading ONNX model with OpenCV DNN...")
        start_load = time.time()
        
        net = cv2.dnn.readNetFromONNX(model_path)
        
        # Set backend to CPU (most compatible)
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        load_time = time.time() - start_load
        logger.info(f"✅ OpenCV DNN model loaded in {load_time:.1f}s")
        
        ML_READY = True
        ML_LOADING = False
        logger.info("🎉 5MB ONNX model fully loaded with OpenCV DNN!")
        return True
        
    except Exception as e:
        error_msg = f"OpenCV model loading failed: {str(e)}"
        logger.error(f"❌ {error_msg}")
        ML_ERROR = error_msg
        ML_LOADING = False
        return False

def preprocess_image(image):
    """Preprocess image for YOLO model"""
    # Resize
    if max(image.size) > MAX_IMAGE_SIZE:
        ratio = MAX_IMAGE_SIZE / max(image.size)
        new_size = tuple(int(dim * ratio) for dim in image.size)
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    # Convert to OpenCV format
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    return img_array, image.size

def postprocess_detections(outputs, original_size, conf_threshold=0.25):
    """Process OpenCV DNN outputs"""
    detections = []
    
    try:
        # OpenCV DNN output format
        for detection in outputs[0]:
            confidence = detection[4]
            
            if confidence > conf_threshold:
                # Get class scores
                scores = detection[5:]
                class_id = np.argmax(scores)
                class_confidence = scores[class_id]
                
                final_confidence = confidence * class_confidence
                
                if final_confidence > conf_threshold:
                    # Extract bounding box
                    x_center, y_center, width, height = detection[:4]
                    
                    detections.append({
                        "className": CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}",
                        "confidence": float(final_confidence),
                        "boundingBox": {
                            "x": float(max(0, x_center - width/2)),
                            "y": float(max(0, y_center - height/2)),
                            "width": float(min(1, width)),
                            "height": float(min(1, height))
                        }
                    })
    
    except Exception as e:
        logger.error(f"❌ Postprocessing error: {e}")
    
    return detections

@app.route('/', methods=['GET'])
def health():
    """Railway health check"""
    uptime = time.time() - start_time
    
    if not ML_READY:
        logger.warning(f"⚠️ Health check - Model not ready (uptime: {uptime:.1f}s)")
        return jsonify({
            "status": "starting",
            "service": "Gym Detection API (OpenCV DNN)",
            "model_ready": False,
            "uptime": round(uptime, 1),
            "error": ML_ERROR
        }), 503
    
    logger.info(f"✅ Health check passed - OpenCV model ready")
    return jsonify({
        "status": "healthy",
        "service": "Gym Detection API",
        "model_ready": True,
        "uptime_seconds": round(uptime, 1),
        "backend": "OpenCV DNN",
        "model_size": "5MB ONNX"
    }), 200

@app.route('/detect', methods=['POST'])
def detect():
    """Detection endpoint using OpenCV DNN"""
    
    if not ML_READY:
        if ML_LOADING:
            return jsonify({
                "success": False,
                "error": "Model loading...",
                "retry_after": 30
            }), 503
        return jsonify({
            "success": False,
            "error": "Model not ready",
            "details": ML_ERROR
        }), 503
    
    try:
        # Get image
        if 'image' in request.files:
            image_data = request.files['image'].read()
        elif request.content_type and 'image' in request.content_type:
            image_data = request.get_data()
        else:
            return jsonify({
                "success": False,
                "error": "No image provided"
            }), 400
        
        if not image_data:
            return jsonify({"success": False, "error": "Empty image"}), 400
        
        # Process image
        try:
            image = Image.open(BytesIO(image_data))
            logger.info(f"📸 Processing image: {image.size}")
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Invalid image: {str(e)}"
            }), 400
        
        # Preprocess
        img_array, original_size = preprocess_image(image)
        
        # Create blob for OpenCV DNN
        blob = cv2.dnn.blobFromImage(
            img_array, 
            1/255.0,  # Scale factor
            (640, 640),  # Size
            (0, 0, 0),  # Mean
            True,  # Swap RB
            crop=False
        )
        
        # Run inference with OpenCV DNN
        start_inference = time.time()
        net.setInput(blob)
        outputs = net.forward()
        inference_time = time.time() - start_inference
        
        # Process results
        detections = postprocess_detections(outputs, original_size)
        
        logger.info(f"✅ OpenCV detection: {len(detections)} objects in {inference_time:.3f}s")
        
        return jsonify({
            "success": True,
            "detections": detections,
            "count": len(detections),
            "metadata": {
                "image_size": list(original_size),
                "inference_time_ms": round(inference_time * 1000, 2),
                "backend": "OpenCV DNN",
                "platform": "Railway"
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Detection error: {e}")
        return jsonify({
            "success": False,
            "error": f"Detection failed: {str(e)}"
        }), 500

@app.errorhandler(413)
def file_too_large(e):
    return jsonify({"success": False, "error": "File too large (max 2MB)"}), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({"success": False, "error": "Not found"}), 404

if __name__ == '__main__':
    logger.info("🚀 Starting Gym Detection API with OpenCV DNN")
    logger.info("📦 Backend: OpenCV DNN (Railway compatible)")
    logger.info("🏋️ Model: 5MB ONNX")
    
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🌐 Port: {port}")
    
    # Pre-load model with OpenCV (no executable stack issues!)
    logger.info("⚡ PRE-LOADING MODEL WITH OPENCV DNN...")
    startup_start = time.time()
    
    if load_opencv_model():
        startup_time = time.time() - startup_start
        logger.info(f"🎉 SUCCESS! OpenCV model loaded in {startup_time:.1f}s")
    else:
        logger.error("❌ OpenCV model loading failed")
        logger.error(f"Error: {ML_ERROR}")
        sys.exit(1)
    
    logger.info(f"🚀 Starting Flask server...")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        threaded=True,
        use_reloader=False
    )