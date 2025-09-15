from flask import Flask, request, jsonify
import os
import time
import json
from io import BytesIO
from PIL import Image
import logging
import requests
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
onnx_session = None

# Configuration
MAX_IMAGE_SIZE = 640
MAX_CONTENT_LENGTH = 5 * 1024 * 1024

# Use your ONNX model from Google Drive
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
]

app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

def load_onnx_model():
    """Load lightweight ONNX model with CPU-only runtime"""
    global ML_READY, ML_LOADING, ML_ERROR, onnx_session
    
    if ML_READY or ML_LOADING:
        return ML_READY
    
    ML_LOADING = True
    logger.info("Starting model loading...")
    
    try:
        # Import ONNX Runtime (should be pre-installed)
        try:
            import onnxruntime as ort
            logger.info(f"ONNX Runtime version: {ort.__version__}")
        except ImportError as e:
            logger.error("ONNX Runtime not found. Make sure it's installed.")
            ML_ERROR = "ONNX Runtime not installed"
            ML_LOADING = False
            return False
        
        # Create CPU-only session options
        sess_options = ort.SessionOptions()
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.log_severity_level = 3  # Reduce logging
        
        # Download ONNX model if not exists
        model_path = "/tmp/model.onnx"
        if not os.path.exists(model_path):
            logger.info("Downloading ONNX model...")
            try:
                response = requests.get(MODEL_URL, timeout=120, stream=True)
                response.raise_for_status()
                
                total_size = 0
                with open(model_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        total_size += len(chunk)
                        
                logger.info(f"ONNX model downloaded: {total_size / 1024 / 1024:.1f}MB")
            except Exception as e:
                logger.error(f"Failed to download model: {e}")
                ML_ERROR = f"Model download failed: {str(e)}"
                ML_LOADING = False
                return False
        else:
            logger.info(f"Using existing model: {os.path.getsize(model_path) / 1024 / 1024:.1f}MB")
        
        # Load ONNX model with CPU provider only
        providers = ['CPUExecutionProvider']
        logger.info("Creating ONNX inference session...")
        
        try:
            onnx_session = ort.InferenceSession(
                model_path, 
                sess_options=sess_options,
                providers=providers
            )
            
            logger.info("✅ CPU-only ONNX Model loaded successfully!")
            logger.info(f"Available providers: {onnx_session.get_providers()}")
            
            # Get model info
            input_info = onnx_session.get_inputs()[0]
            output_info = onnx_session.get_outputs()[0]
            logger.info(f"Model input: {input_info.name} - {input_info.shape}")
            logger.info(f"Model output: {output_info.name} - {output_info.shape}")
            
        except Exception as e:
            logger.error(f"Failed to create ONNX session: {e}")
            ML_ERROR = f"ONNX session creation failed: {str(e)}"
            ML_LOADING = False
            return False
        
        ML_READY = True
        ML_LOADING = False
        return True
        
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        ML_ERROR = str(e)
        ML_LOADING = False
        return False

def preprocess_image(image):
    """Preprocess image for YOLO inference"""
    try:
        original_size = image.size
        logger.info(f"Original image size: {original_size}")
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize while maintaining aspect ratio
        if max(image.size) > MAX_IMAGE_SIZE:
            ratio = MAX_IMAGE_SIZE / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        img_array = np.array(image).astype(np.float32)
        img_array = img_array / 255.0  # Normalize to 0-1
        
        # Pad to square (640x640)
        h, w = img_array.shape[:2]
        
        # Create 640x640 square canvas
        square_img = np.zeros((MAX_IMAGE_SIZE, MAX_IMAGE_SIZE, 3), dtype=np.float32)
        
        # Calculate padding to center the image
        pad_h = (MAX_IMAGE_SIZE - h) // 2
        pad_w = (MAX_IMAGE_SIZE - w) // 2
        
        # Place image in center
        square_img[pad_h:pad_h+h, pad_w:pad_w+w] = img_array
        
        # Transpose to CHW format and add batch dimension
        img_tensor = square_img.transpose(2, 0, 1)  # HWC to CHW
        img_tensor = np.expand_dims(img_tensor, axis=0)  # Add batch dimension
        
        logger.info(f"Preprocessed tensor shape: {img_tensor.shape}")
        return img_tensor, original_size, (pad_w, pad_h, w, h)
        
    except Exception as e:
        logger.error(f"Preprocessing error: {e}")
        raise

def postprocess_detections(outputs, original_size, padding_info, conf_threshold=0.25):
    """Process ONNX model outputs"""
    detections = []
    pad_w, pad_h, orig_w, orig_h = padding_info
    
    try:
        if not outputs or len(outputs) == 0:
            logger.warning("No outputs from model")
            return detections
        
        # Get the main output tensor
        predictions = outputs[0]
        logger.info(f"Model output shape: {predictions.shape}")
        
        # Handle batch dimension
        if len(predictions.shape) == 3:
            predictions = predictions[0]  # Remove batch dimension
        
        # YOLOv5/v8 format: [num_detections, 5 + num_classes]
        # [x_center, y_center, width, height, confidence, class_scores...]
        
        valid_detections = 0
        for i, detection in enumerate(predictions):
            if len(detection) < 5:
                continue
                
            # Extract basic info
            x_center, y_center, width, height, confidence = detection[:5]
            
            if confidence > conf_threshold:
                valid_detections += 1
                
                # Get class prediction
                if len(detection) > 5:
                    class_scores = detection[5:]
                    class_id = int(np.argmax(class_scores))
                    class_confidence = float(class_scores[class_id])
                    final_confidence = float(confidence * class_confidence)
                else:
                    # Single class model
                    class_id = 0
                    final_confidence = float(confidence)
                
                if final_confidence > conf_threshold:
                    # Convert coordinates from model space to original image space
                    # Model outputs are in 640x640 space, need to account for padding
                    
                    # Convert to pixel coordinates in 640x640 space
                    x_center_px = x_center * MAX_IMAGE_SIZE
                    y_center_px = y_center * MAX_IMAGE_SIZE
                    width_px = width * MAX_IMAGE_SIZE
                    height_px = height * MAX_IMAGE_SIZE
                    
                    # Account for padding - translate to original image coordinates
                    x_center_orig = (x_center_px - pad_w) / orig_w
                    y_center_orig = (y_center_px - pad_h) / orig_h
                    width_orig = width_px / orig_w
                    height_orig = height_px / orig_h
                    
                    # Convert center format to top-left corner format
                    x_corner = x_center_orig - width_orig / 2
                    y_corner = y_center_orig - height_orig / 2
                    
                    # Clamp to valid range [0, 1]
                    x_corner = max(0, min(1, x_corner))
                    y_corner = max(0, min(1, y_corner))
                    width_orig = max(0, min(1, width_orig))
                    height_orig = max(0, min(1, height_orig))
                    
                    # Get class name
                    class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else f"class_{class_id}"
                    
                    detections.append({
                        "className": class_name,
                        "confidence": final_confidence,
                        "boundingBox": {
                            "x": float(x_corner),
                            "y": float(y_corner), 
                            "width": float(width_orig),
                            "height": float(height_orig)
                        }
                    })
        
        logger.info(f"Found {valid_detections} detections above confidence threshold")
        logger.info(f"Returning {len(detections)} final detections")
        
    except Exception as e:
        logger.error(f"Postprocessing error: {e}")
        logger.error(f"Outputs info: {[out.shape if hasattr(out, 'shape') else type(out) for out in outputs]}")
    
    return detections

# Flask routes
@app.route('/', methods=['GET'])
def health():
    """Health check endpoint"""
    uptime = time.time() - start_time
    
    return jsonify({
        "status": "healthy",
        "service": "Gym Equipment Detection API",
        "ml_ready": ML_READY,
        "ml_loading": ML_LOADING,
        "ml_error": ML_ERROR,
        "uptime_seconds": round(uptime, 1),
        "model_type": "ONNX CPU-only",
        "endpoints": {
            "health": "/",
            "detect": "/detect (POST)",
            "quick_test": "/quick-test"
        },
        "supported_classes": CLASS_NAMES
    })

@app.route('/quick-test', methods=['GET'])
def quick_test():
    """Quick connectivity test for mobile apps"""
    return jsonify({
        "status": "ok",
        "timestamp": time.time(),
        "server": "responsive",
        "ml_ready": ML_READY
    })

@app.route('/detect', methods=['POST'])
def detect():
    """Main detection endpoint"""
    start_time_request = time.time()
    
    # Check if model is ready
    if not ML_READY:
        if ML_LOADING:
            return jsonify({
                "success": False,
                "error": "Model is still loading, please wait...",
                "retry_after": 30,
                "status": "loading"
            }), 503
        
        # Try to load model
        logger.info("Loading model on first request...")
        if not load_onnx_model():
            return jsonify({
                "success": False,
                "error": "Model initialization failed",
                "details": ML_ERROR,
                "status": "failed"
            }), 500
    
    try:
        # Extract image from request
        image_data = None
        if 'image' in request.files:
            file = request.files['image']
            if file.filename:
                image_data = file.read()
        elif request.content_type and 'image' in request.content_type:
            image_data = request.get_data()
        
        if not image_data:
            return jsonify({
                "success": False,
                "error": "No image data found. Send image as 'image' form field or raw data."
            }), 400
        
        logger.info(f"Received image data: {len(image_data)} bytes")
        
        # Load and validate image
        try:
            image = Image.open(BytesIO(image_data))
            logger.info(f"Image loaded: {image.size}, mode: {image.mode}")
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Invalid image format: {str(e)}"
            }), 400
        
        # Preprocess image
        try:
            processed_image, original_size, padding_info = preprocess_image(image)
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Image preprocessing failed: {str(e)}"
            }), 500
        
        # Run inference
        try:
            input_name = onnx_session.get_inputs()[0].name
            
            inference_start = time.time()
            outputs = onnx_session.run(None, {input_name: processed_image})
            inference_time = time.time() - inference_start
            
            logger.info(f"Inference completed in {inference_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            return jsonify({
                "success": False,
                "error": f"Model inference failed: {str(e)}"
            }), 500
        
        # Post-process results
        try:
            detections = postprocess_detections(outputs, original_size, padding_info)
        except Exception as e:
            logger.error(f"Post-processing failed: {e}")
            return jsonify({
                "success": False,
                "error": f"Results processing failed: {str(e)}"
            }), 500
        
        total_time = time.time() - start_time_request
        
        logger.info(f"✅ Detection completed: {len(detections)} objects found in {total_time:.3f}s")
        
        return jsonify({
            "success": True,
            "detections": detections,
            "count": len(detections),
            "image_size": list(original_size),
            "processing_time": {
                "total_seconds": round(total_time, 3),
                "inference_seconds": round(inference_time, 3)
            },
            "model_info": {
                "type": "ONNX",
                "provider": "CPU",
                "classes": len(CLASS_NAMES)
            }
        })
        
    except Exception as e:
        logger.error(f"Unexpected error in detect endpoint: {e}")
        return jsonify({
            "success": False,
            "error": "Internal server error",
            "details": str(e) if app.debug else None
        }), 500

@app.errorhandler(413)
def file_too_large(e):
    return jsonify({
        "success": False,
        "error": "File too large",
        "max_size": "5MB"
    }), 413

@app.errorhandler(404)
def not_found(e):
    return jsonify({
        "success": False,
        "error": "Endpoint not found",
        "available_endpoints": ["/", "/detect", "/quick-test"]
    }), 404

@app.errorhandler(500)
def internal_error(e):
    logger.error(f"Internal server error: {e}")
    return jsonify({
        "success": False,
        "error": "Internal server error"
    }), 500

if __name__ == '__main__':
    logger.info("🚀 Starting Gym Equipment Detection API")
    logger.info(f"Model URL: {MODEL_URL}")
    logger.info(f"Supported classes: {len(CLASS_NAMES)}")
    
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('DEBUG', 'false').lower() == 'true'
    
    logger.info(f"Server starting on port {port}, debug={debug_mode}")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug_mode,
        threaded=True
    )