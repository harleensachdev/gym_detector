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
    # Resize maintaining aspect ratio
    target_size = 640
    w, h = image.size
    
    # Calculate new dimensions
    if w > h:
        new_w = target_size
        new_h = int(h * target_size / w)
    else:
        new_h = target_size
        new_w = int(w * target_size / h)
    
    # Resize image
    image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    
    # Convert to OpenCV format
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    
    return img_array, (w, h)  # Return original size

def postprocess_detections(outputs, original_size, conf_threshold=0.25):
    """Process OpenCV DNN outputs with proper YOLO format"""
    detections = []
    
    try:
        # OpenCV DNN output handling
        output = outputs[0]
        
        logger.info(f"Raw output shape: {output.shape}")
        
        # Handle YOLO output format - typically (num_features, num_detections)
        # Need to transpose to (num_detections, num_features)
        if output.shape[0] < output.shape[1]:
            output = output.T  # Transpose to get (8400, 18)
        
        logger.info(f"Transposed output shape: {output.shape}")
        logger.info(f"Processing {len(output)} potential detections")
        
        # Calculate actual number of classes from output shape
        num_features = output.shape[1]  # Should be 18
        num_classes = num_features - 5  # 18 - 4 (bbox) - 1 (objectness) = 13
        
        logger.info(f"Detected {num_classes} classes in model output")
        
        # Use only the classes that exist in the model
        available_classes = CLASS_NAMES[:num_classes]
        logger.info(f"Using classes: {available_classes}")
        
        detection_count = 0
        
        # Add debug output for first few detections
        for i, detection in enumerate(output[:5]):  # Check first 5 only for debugging
            logger.info(f"Debug detection {i}: shape={detection.shape}, "
                       f"first_10_values={detection[:10].tolist()}")
        
        for i, detection in enumerate(output):
            if len(detection) < 5:
                continue
                
            # Extract components - YOLO format: [x, y, w, h, objectness, class_scores...]
            x_center, y_center, width, height = detection[:4]
            objectness = detection[4]
            
            # Get class scores (only as many as the model actually outputs)
            if len(detection) > 5:
                class_scores = detection[5:5+num_classes]
            else:
                continue
            
            # Apply objectness threshold
            if objectness > conf_threshold and len(class_scores) > 0:
                # Find best class
                class_id = np.argmax(class_scores)
                class_confidence = class_scores[class_id]
                
                # Combined confidence
                final_confidence = float(objectness * class_confidence)
                
                # Apply final threshold and validate class
                if final_confidence > conf_threshold and 0 <= class_id < len(available_classes):
                    # Normalize confidence to 0-1 range
                    final_confidence = min(1.0, max(0.0, final_confidence))
                    
                    # Convert center coordinates to top-left coordinates
                    # YOLO coordinates are typically already normalized (0-1)
                    x_top_left = max(0.0, min(1.0, float(x_center - width/2)))
                    y_top_left = max(0.0, min(1.0, float(y_center - height/2)))
                    box_width = max(0.0, min(1.0, float(width)))
                    box_height = max(0.0, min(1.0, float(height)))
                    
                    # Ensure box stays within bounds
                    if x_top_left + box_width > 1.0:
                        box_width = 1.0 - x_top_left
                    if y_top_left + box_height > 1.0:
                        box_height = 1.0 - y_top_left
                    
                    # Only add if box has reasonable size
                    if box_width > 0.01 and box_height > 0.01:
                        detection_obj = {
                            "className": available_classes[class_id],
                            "confidence": final_confidence,
                            "boundingBox": {
                                "x": x_top_left,
                                "y": y_top_left,
                                "width": box_width,
                                "height": box_height
                            }
                        }
                        
                        detections.append(detection_obj)
                        detection_count += 1
                        
                        logger.info(f"Detection {detection_count}: {available_classes[class_id]} "
                                  f"conf={final_confidence:.3f} bbox=({x_top_left:.3f}, "
                                  f"{y_top_left:.3f}, {box_width:.3f}, {box_height:.3f})")
                        
                        if detection_count >= 50:  # Limit detections
                            break
    
    except Exception as e:
        logger.error(f"❌ Postprocessing error: {e}")
        logger.error(f"Output shape: {outputs[0].shape if len(outputs) > 0 else 'No outputs'}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
    
    # Sort by confidence and return top detections
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    final_detections = detections[:10]  # Return top 10
    
    logger.info(f"Returning {len(final_detections)} valid detections")
    return final_detections

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

@app.route('/quick-test', methods=['GET'])
def quick_test():
    """Quick test endpoint that iOS app is calling"""
    uptime = time.time() - start_time
    
    if not ML_READY:
        return jsonify({
            "status": "loading",
            "service": "Gym Detection API",
            "model_ready": False,
            "uptime": round(uptime, 1),
            "message": "Model still loading..."
        }), 503
    
    return jsonify({
        "status": "ok",
        "service": "Gym Detection API",
        "model_ready": True,
        "uptime": round(uptime, 1),
        "message": "Quick test successful",
        "classes_available": len(CLASS_NAMES),
        "sample_classes": CLASS_NAMES[:5]
    }), 200

@app.route('/detect', methods=['POST'])
def detect():
    """Detection endpoint using OpenCV DNN"""
    
    if not ML_READY:
        if ML_LOADING:
            return jsonify({
                "success": False,
                "error": "Model loading, please wait...",
                "retry_after": 30
            }), 503
        return jsonify({
            "success": False,
            "error": "Model not ready",
            "details": ML_ERROR
        }), 503
    
    try:
        # Get image from request
        image_data = None
        if 'image' in request.files:
            image_data = request.files['image'].read()
        elif request.content_type and 'image' in request.content_type:
            image_data = request.get_data()
        else:
            return jsonify({
                "success": False,
                "error": "No image provided. Send as 'image' file or raw image data."
            }), 400
        
        if not image_data or len(image_data) == 0:
            return jsonify({
                "success": False, 
                "error": "Empty image data received"
            }), 400
        
        logger.info(f"📸 Processing image of size {len(image_data)} bytes")
        
        # Process image
        try:
            image = Image.open(BytesIO(image_data))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            logger.info(f"Image loaded: {image.size}, mode: {image.mode}")
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Invalid image format: {str(e)}"
            }), 400
        
        # Preprocess image
        img_array, original_size = preprocess_image(image)
        logger.info(f"Image preprocessed to {img_array.shape}")
        
        # Create blob for OpenCV DNN
        blob = cv2.dnn.blobFromImage(
            img_array, 
            1/255.0,          # Scale factor (normalize to 0-1)
            (640, 640),       # Target size
            (0, 0, 0),        # Mean subtraction
            swapRB=True,      # Swap R and B channels
            crop=False        # Don't crop, just resize
        )
        
        logger.info(f"Blob created with shape: {blob.shape}")
        
        # Run inference with OpenCV DNN
        start_inference = time.time()
        net.setInput(blob)
        outputs = net.forward()
        inference_time = time.time() - start_inference
        
        logger.info(f"Inference completed in {inference_time:.3f}s")
        logger.info(f"Raw output shape: {outputs[0].shape if len(outputs) > 0 else 'No outputs'}")
        
        # Process results
        detections = postprocess_detections(outputs, original_size, conf_threshold=0.1)  # Lower threshold for debugging
        
        logger.info(f"✅ Final result: {len(detections)} valid detections")
        
        # Log detection summary
        if detections:
            for i, det in enumerate(detections[:3]):
                logger.info(f"  Top {i+1}: {det['className']} ({det['confidence']:.1%})")
        
        response_data = {
            "success": True,
            "detections": detections,
            "count": len(detections),
            "metadata": {
                "image_size": list(original_size),
                "processed_size": [640, 640],
                "inference_time_ms": round(inference_time * 1000, 2),
                "backend": "OpenCV DNN",
                "platform": "Railway",
                "model_classes": len(CLASS_NAMES),
                "confidence_threshold": 0.3
            }
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        logger.error(f"❌ Detection error: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        
        return jsonify({
            "success": False,
            "error": f"Detection failed: {str(e)}",
            "type": "internal_error"
        }), 500

@app.route('/debug/info', methods=['GET'])
def debug_info():
    """Debug endpoint to check model state"""
    return jsonify({
        "model_ready": ML_READY,
        "model_loading": ML_LOADING,
        "model_error": ML_ERROR,
        "uptime": time.time() - start_time,
        "class_names": CLASS_NAMES,
        "class_count": len(CLASS_NAMES),
        "max_image_size": MAX_IMAGE_SIZE
    })

@app.errorhandler(413)
def file_too_large(e):
    return jsonify({
        "success": False, 
        "error": f"File too large (max {MAX_CONTENT_LENGTH//1024//1024}MB)"
    }), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "success": False, 
        "error": "Endpoint not found",
        "available_endpoints": ["/", "/detect", "/quick-test", "/debug/info"]
    }), 404

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({
        "success": False,
        "error": "Internal server error"
    }), 500

if __name__ == '__main__':
    logger.info("🚀 Starting Gym Detection API with OpenCV DNN")
    logger.info("📦 Backend: OpenCV DNN (Railway compatible)")
    logger.info("🏋️ Model: 5MB ONNX")
    logger.info(f"🎯 Classes: {len(CLASS_NAMES)} gym equipment types")
    
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🌐 Port: {port}")
    
    # Pre-load model with OpenCV
    logger.info("⚡ PRE-LOADING MODEL WITH OPENCV DNN...")
    startup_start = time.time()
    
    if load_opencv_model():
        startup_time = time.time() - startup_start
        logger.info(f"🎉 SUCCESS! OpenCV model loaded in {startup_time:.1f}s")
        logger.info(f"📋 Available classes: {', '.join(CLASS_NAMES[:5])}...")
    else:
        logger.error("❌ OpenCV model loading failed")
        logger.error(f"Error: {ML_ERROR}")
        sys.exit(1)
    
    logger.info(f"🚀 Starting Flask server on 0.0.0.0:{port}")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        threaded=True,
        use_reloader=False
    )