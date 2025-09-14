from flask import Flask, request, jsonify
import os
import sys
from io import BytesIO
from PIL import Image
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global model variable
model = None
MODEL_LOADED = False

def load_model():
    """Load YOLO model with error handling"""
    global model, MODEL_LOADED
    
    try:
        # Try to import ultralytics
        from ultralytics import YOLO
        logger.info("Successfully imported ultralytics")
        
        # Look for model files in order of preference
        model_files = [
            "model_epoch_85.pt"
        ]
        
        model_path = None
        for model_file in model_files:
            if os.path.exists(model_file):
                model_path = model_file
                logger.info(f"Found model file: {model_file}")
                break
        
        if model_path:
            logger.info(f"Loading model from {model_path}")
            model = YOLO(model_path)
            MODEL_LOADED = True
            logger.info(f"Model loaded successfully with {len(model.names)} classes")
            logger.info(f"Classes: {list(model.names.values())}")
        else:
            logger.warning("No model file found, using YOLOv8n as fallback")
            model = YOLO('yolov8n.pt')  # This will download automatically
            MODEL_LOADED = True
            logger.info("Fallback model loaded successfully")
            
    except ImportError as e:
        logger.error(f"Failed to import required packages: {e}")
        MODEL_LOADED = False
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        MODEL_LOADED = False

# Try to load model on startup
logger.info("Starting model loading...")
load_model()

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": MODEL_LOADED,
        "python_version": sys.version,
        "available_classes": list(model.names.values()) if MODEL_LOADED and model else [],
        "message": "YOLO Gym Equipment Detection API" if MODEL_LOADED else "Model loading failed - check logs"
    })

@app.route('/status', methods=['GET'])
def detailed_status():
    """Detailed status for debugging"""
    status_info = {
        "model_loaded": MODEL_LOADED,
        "python_version": sys.version,
        "working_directory": os.getcwd(),
        "files_in_directory": os.listdir('.'),
    }
    
    if MODEL_LOADED and model:
        status_info["num_classes"] = len(model.names)
        status_info["class_names"] = list(model.names.values())
    
    # Check for common issues
    try:
        import torch
        status_info["torch_version"] = torch.__version__
        status_info["cuda_available"] = torch.cuda.is_available()
    except:
        status_info["torch_available"] = False
        
    try:
        import cv2
        status_info["opencv_version"] = cv2.__version__
    except:
        status_info["opencv_available"] = False
        
    return jsonify(status_info)

@app.route('/detect', methods=['POST'])
def detect_objects():
    """Main detection endpoint with comprehensive error handling"""
    if not MODEL_LOADED or model is None:
        return jsonify({
            "success": False,
            "error": "Model not loaded. Check /status endpoint for details."
        }), 500
    
    try:
        # Get image data
        if 'image' in request.files:
            file = request.files['image']
            image_data = file.read()
            logger.info(f"Received file upload: {file.filename}, size: {len(image_data)} bytes")
        else:
            image_data = request.get_data()
            logger.info(f"Received raw image data, size: {len(image_data)} bytes")

        if not image_data or len(image_data) < 100:  # Minimum reasonable image size
            return jsonify({
                "success": False,
                "error": "No valid image data provided"
            }), 400

        # Convert to PIL Image
        try:
            image = Image.open(BytesIO(image_data))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            logger.info(f"Image loaded: {image.size}, mode: {image.mode}")
        except Exception as e:
            logger.error(f"Failed to decode image: {e}")
            return jsonify({
                "success": False,
                "error": f"Invalid image format: {str(e)}"
            }), 400

        # Run inference
        logger.info("Running YOLO inference...")
        results = model(image, verbose=False)
        logger.info(f"Inference completed, {len(results)} result objects")

        # Parse results
        detections = []
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                logger.info(f"Found {len(result.boxes)} detections")
                
                for box in result.boxes:
                    # Get normalized coordinates
                    x1, y1, x2, y2 = box.xyxyn[0].tolist()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    
                    # Convert to center format
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1
                    
                    detection = {
                        "className": class_name,
                        "confidence": confidence,
                        "boundingBox": {
                            "x": center_x - width/2,
                            "y": center_y - height/2,
                            "width": width,
                            "height": height
                        }
                    }
                    
                    detections.append(detection)
                    logger.info(f"Detection: {class_name} ({confidence:.2f})")
            else:
                logger.info("No detections found in this result")

        logger.info(f"Returning {len(detections)} total detections")
        return jsonify({
            "success": True,
            "detections": detections,
            "count": len(detections),
            "model_classes": list(model.names.values())
        })

    except Exception as e:
        logger.error(f"Detection error: {str(e)}", exc_info=True)
        return jsonify({
            "success": False,
            "error": f"Detection failed: {str(e)}",
            "error_type": type(e).__name__
        }), 500

@app.route('/classes', methods=['GET'])
def get_classes():
    """Get available classes"""
    if not MODEL_LOADED or model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    return jsonify({
        "classes": list(model.names.values()),
        "num_classes": len(model.names)
    })

@app.route('/test', methods=['GET'])
def test_model():
    """Test endpoint with a synthetic image"""
    if not MODEL_LOADED or model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        # Create a simple test image
        test_image = Image.new('RGB', (640, 640), color='red')
        
        # Run detection
        results = model(test_image, verbose=False)
        
        detection_count = 0
        for result in results:
            if result.boxes is not None:
                detection_count += len(result.boxes)
        
        return jsonify({
            "success": True,
            "message": "Model test completed",
            "detections_found": detection_count,
            "model_classes": list(model.names.values())
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Test failed: {str(e)}"
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    
    logger.info(f"Starting Flask app on port {port}")
    logger.info(f"Model loaded: {MODEL_LOADED}")
    
    app.run(
        host='0.0.0.0', 
        port=port, 
        debug=debug_mode
    )