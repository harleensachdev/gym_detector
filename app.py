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

# Configuration - Updated to 320x320 as per your training
INPUT_SIZE = 320  # Changed from 640 to 320
MAX_CONTENT_LENGTH = 2 * 1024 * 1024

# Your ONNX model URL
MODEL_URL = os.getenv("MODEL_URL", "https://drive.google.com/uc?export=download&id=10x_l-SAe3uuoGjIK_wZIOhQ0WBTx79i7")
CLASS_NAMES = [
    "arm_curl", "chest_fly", "chest_press", "chin_dip", "dumbell",
    "lat_pulldown", "lateral_raises", "leg_curl", "leg_extension",
    "leg_press", "seated_cable_row", "seated_dip", "shoulder_press", "smith_machine"
]

app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

def letterbox_image(image, target_size=(320, 320), color=(114, 114, 114)):
    """
    Resize image with unchanged aspect ratio using padding - EXACTLY like Ultralytics
    Updated for 320x320 input size
    """
    # Get current image dimensions
    h, w = image.shape[:2]
    target_h, target_w = target_size
    
    # Calculate scaling ratio (same for both dimensions to maintain aspect ratio)
    ratio = min(target_w / w, target_h / h)
    
    # Calculate new dimensions after scaling
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    
    # Resize the image
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    
    # Create a new image with target size filled with padding color
    padded = np.full((target_h, target_w, 3), color, dtype=np.uint8)
    
    # Calculate padding offsets to center the image
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    
    # Place the resized image in the center of the padded image
    padded[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    
    logger.info(f"Letterboxed: {(w, h)} -> {(new_w, new_h)} -> {target_size}, ratio={ratio:.3f}")
    
    return padded, ratio, (pad_x, pad_y)

def preprocess_image_ultralytics_style(image):
    """
    Preprocess image EXACTLY like Ultralytics YOLO does for 320x320
    """
    # Convert PIL to OpenCV format if needed
    if isinstance(image, Image.Image):
        img_array = np.array(image)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    else:
        img_array = image.copy()
    
    # Apply letterbox resize for 320x320
    letterboxed, ratio, padding = letterbox_image(img_array, target_size=(INPUT_SIZE, INPUT_SIZE))
    
    logger.info(f"Original shape: {img_array.shape}")
    logger.info(f"Letterboxed shape: {letterboxed.shape}")
    
    return letterboxed, ratio, padding, img_array.shape[:2]

def create_ultralytics_blob(img_array):
    """
    Create blob exactly like Ultralytics does for 320x320
    """
    # Convert BGR to RGB (Ultralytics expects RGB)
    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    
    # Create blob: HWC -> CHW, normalize to 0-1, add batch dimension
    blob = cv2.dnn.blobFromImage(
        img_rgb,
        scalefactor=1.0/255.0,  # Normalize to 0-1
        size=(INPUT_SIZE, INPUT_SIZE),  # 320x320
        mean=(0, 0, 0),         # No mean subtraction
        swapRB=False,           # Already RGB, don't swap
        crop=False,             # Don't crop (already letterboxed)
        ddepth=cv2.CV_32F       # Use float32
    )
    
    logger.info(f"Blob shape: {blob.shape}")
    logger.info(f"Blob dtype: {blob.dtype}")
    logger.info(f"Blob range: [{blob.min():.6f}, {blob.max():.6f}]")
    
    return blob

def postprocess_ultralytics_style(outputs, original_shape, ratio, padding, conf_threshold=0.25):
    """
    Postprocess outputs EXACTLY like Ultralytics does for 320x320 model
    """
    detections = []
    
    try:
        # Get the output tensor
        output = outputs[0]  # Shape varies by model
        
        logger.info(f"Raw output shape: {output.shape}")
        
        # Handle different output formats
        if len(output.shape) == 3:
            if output.shape[0] == 1:
                output = output[0]  # Remove batch dim: (features, detections)
            else:
                logger.info(f"Unexpected 3D shape: {output.shape}")
        
        # Ensure we have (num_detections, num_features) format
        if len(output.shape) == 2:
            if output.shape[0] < output.shape[1]:
                output = output.T  # Transpose to (detections, features)
        
        logger.info(f"Processed output shape: {output.shape}")
        
        if len(output.shape) != 2:
            logger.error(f"Cannot process output shape: {output.shape}")
            return []
        
        num_detections, num_features = output.shape
        logger.info(f"Processing {num_detections} detections with {num_features} features each")
        
        # Determine the format based on number of features
        if num_features == len(CLASS_NAMES) + 4:
            # YOLOv8 format: [x, y, w, h, class_0, class_1, ..., class_n]
            boxes = output[:, :4]
            class_confidences = output[:, 4:]
            logger.info("Detected YOLOv8 format (no objectness)")
        elif num_features == len(CLASS_NAMES) + 5:
            # YOLOv5 format: [x, y, w, h, objectness, class_0, class_1, ..., class_n]
            boxes = output[:, :4]
            objectness = output[:, 4]
            class_confidences = output[:, 5:]
            logger.info("Detected YOLOv5 format (with objectness)")
        else:
            # Try to auto-detect
            expected_classes = num_features - 4  # Assume no objectness
            if expected_classes > 0 and expected_classes <= len(CLASS_NAMES):
                boxes = output[:, :4]
                class_confidences = output[:, 4:]
                logger.info(f"Auto-detected format: {expected_classes} classes, no objectness")
            else:
                expected_classes = num_features - 5  # Assume with objectness
                if expected_classes > 0 and expected_classes <= len(CLASS_NAMES):
                    boxes = output[:, :4]
                    objectness = output[:, 4]
                    class_confidences = output[:, 5:]
                    logger.info(f"Auto-detected format: {expected_classes} classes, with objectness")
                else:
                    logger.error(f"Cannot determine output format. Features: {num_features}, Expected classes: {len(CLASS_NAMES)}")
                    return []
        
        logger.info(f"Boxes shape: {boxes.shape}")
        logger.info(f"Class confidences shape: {class_confidences.shape}")
        logger.info(f"Max class confidence: {class_confidences.max():.6f}")
        logger.info(f"Mean class confidence: {class_confidences.mean():.6f}")
        
        # Use only available classes
        available_classes = min(class_confidences.shape[1], len(CLASS_NAMES))
        logger.info(f"Using {available_classes} classes")
        
        # Find detections above threshold
        valid_detections = 0
        for i in range(num_detections):
            # Get class confidences for this detection
            class_scores = class_confidences[i, :available_classes]
            max_conf = np.max(class_scores)
            class_id = np.argmax(class_scores)
            
            # Apply objectness if available
            final_confidence = max_conf
            if 'objectness' in locals():
                final_confidence = max_conf * objectness[i]
            
            # Check confidence threshold
            if final_confidence >= conf_threshold:
                # Extract box coordinates (in normalized space relative to input size)
                x_center, y_center, width, height = boxes[i]
                
                # Convert from center format to corner format
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                x2 = x_center + width / 2
                y2 = y_center + height / 2
                
                # Adjust for letterbox padding and scaling
                pad_x, pad_y = padding
                
                # Remove padding (convert from INPUT_SIZE space to original space)
                x1 = (x1 * INPUT_SIZE - pad_x) / ratio
                y1 = (y1 * INPUT_SIZE - pad_y) / ratio
                x2 = (x2 * INPUT_SIZE - pad_x) / ratio
                y2 = (y2 * INPUT_SIZE - pad_y) / ratio
                
                # Normalize to original image size
                orig_h, orig_w = original_shape
                x1 /= orig_w
                y1 /= orig_h
                x2 /= orig_w
                y2 /= orig_h
                
                # Clamp to valid range
                x1 = max(0.0, min(1.0, x1))
                y1 = max(0.0, min(1.0, y1))
                x2 = max(0.0, min(1.0, x2))
                y2 = max(0.0, min(1.0, y2))
                
                # Calculate final width and height
                box_width = x2 - x1
                box_height = y2 - y1
                
                # Only keep reasonable sized boxes
                if box_width > 0.01 and box_height > 0.01 and 0 <= class_id < available_classes:
                    detection = {
                        "className": CLASS_NAMES[class_id],
                        "confidence": float(final_confidence),
                        "boundingBox": {
                            "x": float(x1),
                            "y": float(y1),
                            "width": float(box_width),
                            "height": float(box_height)
                        }
                    }
                    
                    detections.append(detection)
                    valid_detections += 1
                    
                    if valid_detections <= 5:  # Log first 5
                        logger.info(f"Detection {valid_detections}: {CLASS_NAMES[class_id]} "
                                   f"conf={final_confidence:.3f} "
                                   f"box=({x1:.3f},{y1:.3f},{box_width:.3f},{box_height:.3f})")
                    
                    if valid_detections >= 50:  # Limit total detections
                        break
    
    except Exception as e:
        logger.error(f"❌ Postprocessing error: {e}")
        import traceback
        logger.error(traceback.format_exc())
    
    # Sort by confidence and apply basic NMS if needed
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Simple NMS to remove overlapping detections
    filtered_detections = []
    for det in detections:
        is_duplicate = False
        for existing in filtered_detections:
            if det['className'] == existing['className']:
                # Calculate IoU
                iou = calculate_iou(det['boundingBox'], existing['boundingBox'])
                if iou > 0.5:  # Remove if IoU > 0.5
                    is_duplicate = True
                    break
        
        if not is_duplicate:
            filtered_detections.append(det)
        
        if len(filtered_detections) >= 10:  # Limit to top 10
            break
    
    logger.info(f"Found {len(detections)} raw detections, {len(filtered_detections)} after NMS")
    
    return filtered_detections

def calculate_iou(box1, box2):
    """Calculate Intersection over Union of two bounding boxes"""
    try:
        # Extract coordinates
        x1_1, y1_1 = box1['x'], box1['y']
        x2_1, y2_1 = x1_1 + box1['width'], y1_1 + box1['height']
        
        x1_2, y1_2 = box2['x'], box2['y']
        x2_2, y2_2 = x1_2 + box2['width'], y1_2 + box2['height']
        
        # Calculate intersection
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        area1 = box1['width'] * box1['height']
        area2 = box2['width'] * box2['height']
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    except:
        return 0.0

def detect_ultralytics_style(image_data, net, conf_threshold=0.25):
    """
    Main detection function that mimics Ultralytics exactly for 320x320
    """
    try:
        # Load image
        image = Image.open(BytesIO(image_data))
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        logger.info(f"📷 Input image: {image.size}")
        
        # Preprocess exactly like Ultralytics for 320x320
        processed_img, ratio, padding, original_shape = preprocess_image_ultralytics_style(image)
        
        # Create blob
        blob = create_ultralytics_blob(processed_img)
        
        # Run inference
        start_time = time.time()
        net.setInput(blob)
        outputs = net.forward()
        inference_time = time.time() - start_time
        
        logger.info(f"⚡ Inference completed in {inference_time:.3f}s")
        
        # Postprocess exactly like Ultralytics
        detections = postprocess_ultralytics_style(
            outputs, original_shape, ratio, padding, conf_threshold
        )
        
        return detections, inference_time
        
    except Exception as e:
        logger.error(f"❌ Detection failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return [], 0.0

def load_opencv_model():
    """Load ONNX model using OpenCV DNN - Railway compatible!"""
    global ML_READY, ML_LOADING, ML_ERROR, net
    
    if ML_READY or ML_LOADING:
        return ML_READY
    
    ML_LOADING = True
    logger.info(f"🚀 Loading ONNX model with OpenCV DNN (input size: {INPUT_SIZE}x{INPUT_SIZE})...")
    
    try:
        # Download model
        model_path = "/tmp/model.onnx"
        if not os.path.exists(model_path):
            logger.info("📥 Downloading model from Google Drive...")
            start_download = time.time()
            
            response = requests.get(
                MODEL_URL, 
                timeout=60,
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
        
        # Load with OpenCV DNN
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
        logger.info(f"🎉 Model ready! Input size: {INPUT_SIZE}x{INPUT_SIZE}, Classes: {len(CLASS_NAMES)}")
        return True
        
    except Exception as e:
        error_msg = f"Model loading failed: {str(e)}"
        logger.error(f"❌ {error_msg}")
        ML_ERROR = error_msg
        ML_LOADING = False
        return False

@app.route('/', methods=['GET'])
def health():
    """Railway health check"""
    uptime = time.time() - start_time
    
    if not ML_READY:
        logger.warning(f"⚠️ Health check - Model not ready (uptime: {uptime:.1f}s)")
        return jsonify({
            "status": "starting",
            "service": "Gym Detection API (320x320)",
            "model_ready": False,
            "uptime": round(uptime, 1),
            "error": ML_ERROR
        }), 503
    
    logger.info(f"✅ Health check passed")
    return jsonify({
        "status": "healthy",
        "service": "Gym Detection API",
        "model_ready": True,
        "uptime_seconds": round(uptime, 1),
        "backend": "OpenCV DNN",
        "input_size": f"{INPUT_SIZE}x{INPUT_SIZE}",
        "classes": len(CLASS_NAMES)
    }), 200

@app.route('/quick-test', methods=['GET'])
def quick_test():
    """Quick test endpoint"""
    uptime = time.time() - start_time
    
    if not ML_READY:
        return jsonify({
            "status": "loading",
            "model_ready": False,
            "uptime": round(uptime, 1)
        }), 503
    
    return jsonify({
        "status": "ok",
        "model_ready": True,
        "uptime": round(uptime, 1),
        "input_size": f"{INPUT_SIZE}x{INPUT_SIZE}",
        "classes": CLASS_NAMES
    }), 200

@app.route('/detect', methods=['POST'])
def detect():
    """Detection endpoint using Ultralytics-compatible processing"""
    
    if not ML_READY:
        return jsonify({
            "success": False,
            "error": "Model not ready",
            "details": ML_ERROR
        }), 503
    
    try:
        # Get image data
        image_data = None
        if 'image' in request.files:
            image_data = request.files['image'].read()
        elif request.content_type and 'image' in request.content_type:
            image_data = request.get_data()
        else:
            return jsonify({
                "success": False,
                "error": "No image provided. Send as 'image' file or raw data."
            }), 400
        
        if not image_data or len(image_data) == 0:
            return jsonify({
                "success": False,
                "error": "Empty image data received"
            }), 400
        
        logger.info(f"📸 Processing image: {len(image_data)} bytes")
        
        # Use Ultralytics-compatible detection
        detections, inference_time = detect_ultralytics_style(
            image_data, 
            net, 
            conf_threshold=0.25
        )
        
        logger.info(f"✅ Detection complete: {len(detections)} objects found")
        
        # Log top detections
        for i, det in enumerate(detections[:3]):
            logger.info(f"  #{i+1}: {det['className']} ({det['confidence']:.1%})")
        
        response_data = {
            "success": True,
            "detections": detections,
            "count": len(detections),
            "metadata": {
                "inference_time_ms": round(inference_time * 1000, 2),
                "backend": "OpenCV DNN",
                "platform": "Railway",
                "input_size": f"{INPUT_SIZE}x{INPUT_SIZE}",
                "model_classes": len(CLASS_NAMES),
                "confidence_threshold": 0.25,
                "processing": "Ultralytics-compatible"
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
    """Debug endpoint"""
    return jsonify({
        "model_ready": ML_READY,
        "model_loading": ML_LOADING,
        "model_error": ML_ERROR,
        "uptime": time.time() - start_time,
        "input_size": f"{INPUT_SIZE}x{INPUT_SIZE}",
        "class_names": CLASS_NAMES,
        "class_count": len(CLASS_NAMES),
        "processing": "Ultralytics-compatible"
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
    logger.info("🚀 Starting Gym Detection API with Ultralytics-Compatible Processing")
    logger.info(f"📦 Backend: OpenCV DNN")
    logger.info(f"🎯 Input Size: {INPUT_SIZE}x{INPUT_SIZE} (matching your training)")
    logger.info(f"🏋️ Classes: {len(CLASS_NAMES)} gym equipment types")
    
    port = int(os.environ.get('PORT', 5000))
    logger.info(f"🌐 Port: {port}")
    
    # Pre-load model
    logger.info("⚡ PRE-LOADING MODEL...")
    startup_start = time.time()
    
    if load_opencv_model():
        startup_time = time.time() - startup_start
        logger.info(f"🎉 SUCCESS! Model loaded in {startup_time:.1f}s")
        logger.info(f"📋 Available classes: {', '.join(CLASS_NAMES[:5])}...")
    else:
        logger.error("❌ Model loading failed")
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