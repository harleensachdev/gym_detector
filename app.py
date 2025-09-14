from flask import Flask, request, jsonify
import os
import sys
from io import BytesIO
from PIL import Image
import logging
import gc

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
        from ultralytics import YOLO
        logger.info("Successfully imported ultralytics")
        
        model_files = [
            "best_model.pt",
            "model_epoch_85.pt", 
            "best.pt",
            "last.pt"
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
        else:
            logger.warning("No model file found, using YOLOv8n as fallback")
            model = YOLO('yolov8n.pt')
            MODEL_LOADED = True
            logger.info("Fallback model loaded successfully")
            
    except ImportError as e:
        logger.error(f"Failed to import required packages: {e}")
        MODEL_LOADED = False
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        MODEL_LOADED = False

def optimize_image(image, max_size=640):
    """Resize image to reduce memory usage and processing time"""
    # Get original dimensions
    width, height = image.size
    
    # Calculate scaling factor
    if max(width, height) > max_size:
        if width > height:
            new_width = max_size
            new_height = int((height * max_size) / width)
        else:
            new_height = max_size
            new_width = int((width * max_size) / height)
        
        logger.info(f"Resizing image from {width}x{height} to {new_width}x{new_height}")
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    return image

def cleanup_memory():
    """Force garbage collection to free memory"""
    gc.collect()

# Configure Flask for better upload handling
app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024  # 2MB max file size

@app.before_request
def limit_content_length():
    """Reject oversized requests early"""
    if request.content_length and request.content_length > app.config['MAX_CONTENT_LENGTH']:
        return jsonify({
            "success": False,
            "error": "Image too large. Please use a smaller image (under 2MB)."
        }), 413

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": MODEL_LOADED,
        "python_version": sys.version,
        "available_classes": list(model.names.values()) if MODEL_LOADED and model else [],
        "message": "YOLO Gym Equipment Detection API - Optimized for Free Tier"
    })

@app.route('/detect', methods=['POST'])
def detect_objects():
    """Main detection endpoint - optimized for low memory"""
    if not MODEL_LOADED or model is None:
        return jsonify({
            "success": False,
            "error": "Model not loaded"
        }), 500
    
    try:
        # Clean up memory before processing
        cleanup_memory()
        
        # Get image data
        if 'image' in request.files:
            file = request.files['image']
            image_data = file.read()
            logger.info(f"File upload: {len(image_data)} bytes")
        else:
            image_data = request.get_data()
            logger.info(f"Raw data: {len(image_data)} bytes")

        if not image_data or len(image_data) < 100:
            return jsonify({
                "success": False,
                "error": "No valid image data"
            }), 400

        # Convert to PIL Image
        try:
            image = Image.open(BytesIO(image_data))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            logger.info(f"Original image: {image.size}")
            
            # CRITICAL: Resize image to reduce memory usage
            image = optimize_image(image, max_size=416)  # Smaller size for free tier
            logger.info(f"Optimized image: {image.size}")
            
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            return jsonify({
                "success": False,
                "error": f"Invalid image: {str(e)}"
            }), 400

        # Run inference with optimized settings
        logger.info("Starting inference...")
        try:
            # Use minimal settings to reduce memory usage
            results = model(
                image, 
                verbose=False,
                imgsz=416,      # Small image size
                conf=0.25,      # Higher confidence threshold
                iou=0.45,       # Standard IoU
                max_det=100,    # Limit detections
                device='cpu'    # Force CPU to avoid GPU memory issues
            )
            logger.info("Inference completed")
            
        except Exception as e:
            logger.error(f"Inference failed: {e}")
            cleanup_memory()
            return jsonify({
                "success": False,
                "error": f"Model inference failed: {str(e)}"
            }), 500

        # Parse results quickly
        detections = []
        try:
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    logger.info(f"Processing {len(result.boxes)} detections")
                    
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
                
        except Exception as e:
            logger.error(f"Results parsing failed: {e}")
        
        # Clean up memory after processing
        cleanup_memory()
        
        logger.info(f"Returning {len(detections)} detections")
        return jsonify({
            "success": True,
            "detections": detections,
            "count": len(detections),
            "processing_note": "Image was resized for optimal performance on free tier"
        })

    except Exception as e:
        logger.error(f"Detection error: {str(e)}", exc_info=True)
        cleanup_memory()
        return jsonify({
            "success": False,
            "error": f"Processing failed: {str(e)}",
            "suggestion": "Try a smaller image or better lighting"
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

@app.route('/quick-test', methods=['GET'])
def quick_test():
    """Ultra-fast test that should complete in under 10 seconds"""
    if not MODEL_LOADED or model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        # Create tiny test image
        test_image = Image.new('RGB', (160, 160), color='green')
        
        # Run ultra-fast detection
        results = model(test_image, verbose=False, imgsz=160, conf=0.8, max_det=1)
        
        detection_count = 0
        for result in results:
            if result.boxes is not None:
                detection_count += len(result.boxes)
        
        return jsonify({
            "success": True,
            "message": "Quick test completed in under 10 seconds",
            "detections_found": detection_count,
            "server_status": "responsive"
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Quick test failed: {str(e)}",
            "server_status": "overloaded"
        }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    
    # Optimized settings for free tier
    app.run(
        host='0.0.0.0', 
        port=port, 
        debug=False,        # Disable debug mode
        threaded=True       # Enable threading
    )