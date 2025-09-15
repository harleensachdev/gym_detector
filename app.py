from flask import Flask, request, jsonify
import os
import sys
import time
import threading
import gc
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

# Global model variables
model = None
MODEL_LOADED = False
MODEL_LOADING = False
MODEL_ERROR = None
start_time = time.time()

# Configuration for Railway
MODEL_CACHE_DIR = "/tmp/models"
MODEL_URL = os.getenv("PRIMARY_MODEL_URL", "https://drive.google.com/uc?export=download&id=1A9z88pRkkWwdF0LNUMxlVq11kG8nNI60")  # Your Google Drive direct link
BACKUP_MODEL_URL = os.getenv("BACKUP_MODEL_URL", "https://drive.google.com/file/d/187Mtm9KMc8mFBRjJcvvOWIlr0R0EEGvl/view?usp=sharing")
MAX_IMAGE_SIZE = 640
MAX_CONTENT_LENGTH = 5 * 1024 * 1024  # 5MB max file size

app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

def ensure_model_dir():
    """Create model cache directory"""
    os.makedirs(MODEL_CACHE_DIR, exist_ok=True)
    logger.info(f"Model cache directory: {MODEL_CACHE_DIR}")

def download_model_simple(url, filename):
    """Simple model download for small 5MB model"""
    if not url:
        logger.warning(f"No URL provided for {filename}")
        return False
    
    filepath = os.path.join(MODEL_CACHE_DIR, filename)
    
    # Check if already exists
    if os.path.exists(filepath) and os.path.getsize(filepath) > 1024 * 1024:  # > 1MB
        logger.info(f"Model {filename} already cached")
        return True
    
    logger.info(f"Downloading {filename} from Google Drive...")
    
    try:
        # Simple download for small files
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        with open(filepath, 'wb') as f:
            f.write(response.content)
        
        file_size = os.path.getsize(filepath)
        if file_size > 1024 * 1024:  # At least 1MB
            logger.info(f"✅ Downloaded {filename} successfully ({file_size / 1024 / 1024:.1f}MB)")
            return True
        else:
            logger.error(f"Downloaded file too small: {file_size} bytes")
            os.remove(filepath)
            return False
        
    except Exception as e:
        logger.error(f"Download failed: {e}")
        if os.path.exists(filepath):
            os.remove(filepath)
        return False

def load_model():
    """Load YOLO model from Google Drive"""
    global model, MODEL_LOADED, MODEL_LOADING, MODEL_ERROR
    
    if MODEL_LOADED or MODEL_LOADING:
        return
    
    MODEL_LOADING = True
    MODEL_ERROR = None
    logger.info("🚀 Starting model loading...")
    
    try:
        from ultralytics import YOLO
        logger.info("✅ Ultralytics imported successfully")
        
        ensure_model_dir()
        model_loaded = False
        
        # Try primary model from Google Drive
        if MODEL_URL:
            logger.info("Trying primary model from Google Drive...")
            if download_model_simple(MODEL_URL, "custom_model.pt"):
                try:
                    model_path = os.path.join(MODEL_CACHE_DIR, "custom_model.pt")
                    model = YOLO(model_path)
                    logger.info(f"✅ Primary model loaded! Classes: {len(model.names)}")
                    logger.info(f"Available classes: {list(model.names.values())}")
                    model_loaded = True
                except Exception as e:
                    logger.error(f"Failed to load primary model: {e}")
        
        # Try backup model if primary failed
        if not model_loaded and BACKUP_MODEL_URL:
            logger.info("Trying backup model...")
            if download_model_simple(BACKUP_MODEL_URL, "backup_model.pt"):
                try:
                    model_path = os.path.join(MODEL_CACHE_DIR, "backup_model.pt")
                    model = YOLO(model_path)
                    logger.info(f"✅ Backup model loaded! Classes: {len(model.names)}")
                    model_loaded = True
                except Exception as e:
                    logger.error(f"Failed to load backup model: {e}")
        
        # Fallback to YOLOv8n
        if not model_loaded:
            logger.info("Loading fallback YOLOv8n...")
            try:
                model = YOLO('yolov8n.pt')  # Auto-downloads
                logger.info("✅ Fallback YOLOv8n loaded")
                model_loaded = True
            except Exception as e:
                logger.error(f"Even fallback failed: {e}")
                MODEL_ERROR = f"All model loading failed: {e}"
                return
        
        # Quick model warmup
        if model_loaded:
            logger.info("Warming up model...")
            dummy_img = Image.new('RGB', (320, 320), color='red')
            _ = model(dummy_img, verbose=False, imgsz=320, conf=0.5, max_det=1)
            logger.info("✅ Model warmup complete!")
            MODEL_LOADED = True
        
    except ImportError as e:
        MODEL_ERROR = f"Import failed: {e}"
        logger.error(MODEL_ERROR)
    except Exception as e:
        MODEL_ERROR = f"Loading failed: {e}"
        logger.error(MODEL_ERROR, exc_info=True)
    finally:
        MODEL_LOADING = False
        gc.collect()

def optimize_image(image):
    """Optimize image for processing"""
    width, height = image.size
    
    # Resize if too large
    if max(width, height) > MAX_IMAGE_SIZE:
        if width > height:
            new_width = MAX_IMAGE_SIZE
            new_height = int((height * MAX_IMAGE_SIZE) / width)
        else:
            new_height = MAX_IMAGE_SIZE
            new_width = int((width * MAX_IMAGE_SIZE) / height)
        
        logger.info(f"Resizing: {width}x{height} → {new_width}x{new_height}")
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Ensure RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return image

# Flask routes
@app.route('/', methods=['GET'])
def health():
    """Health check"""
    uptime = time.time() - start_time
    
    return jsonify({
        "status": "healthy" if MODEL_LOADED else ("loading" if MODEL_LOADING else "error"),
        "model_loaded": MODEL_LOADED,
        "model_loading": MODEL_LOADING,
        "error": MODEL_ERROR,
        "uptime": round(uptime, 1),
        "classes": list(model.names.values()) if MODEL_LOADED and model else [],
        "message": "YOLO Detection API - Railway + Google Drive"
    })

@app.route('/load-model', methods=['POST'])
def load_model_endpoint():
    """Manually trigger model loading"""
    if MODEL_LOADED:
        return jsonify({
            "success": True,
            "message": "Model already loaded",
            "classes": list(model.names.values())
        })
    
    if MODEL_LOADING:
        return jsonify({
            "success": False,
            "message": "Model is already loading..."
        }), 202
    
    # Start loading in background
    threading.Thread(target=load_model, daemon=True).start()
    
    return jsonify({
        "success": True,
        "message": "Model loading started"
    }), 202

@app.route('/detect', methods=['POST'])
def detect():
    """Main detection endpoint"""
    
    # Auto-load model if needed
    if not MODEL_LOADED and not MODEL_LOADING:
        logger.info("Auto-starting model load...")
        threading.Thread(target=load_model, daemon=True).start()
    
    # Check model status
    if MODEL_LOADING:
        return jsonify({
            "success": False,
            "error": "Model still loading, please wait...",
            "retry_after": 10
        }), 503
    
    if not MODEL_LOADED:
        return jsonify({
            "success": False,
            "error": "Model not loaded",
            "details": MODEL_ERROR,
            "suggestion": "Try calling /load-model first"
        }), 500
    
    try:
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

        # Process image
        try:
            image = Image.open(BytesIO(image_data))
            original_size = image.size
            image = optimize_image(image)
            logger.info(f"Image processed: {original_size} → {image.size}")
        except Exception as e:
            return jsonify({
                "success": False,
                "error": f"Invalid image: {str(e)}"
            }), 400

        # Run detection
        logger.info("Running detection...")
        start_inference = time.time()
        
        results = model(
            image,
            verbose=False,
            imgsz=640,
            conf=0.25,
            iou=0.45,
            max_det=100,
            device='cpu'
        )
        
        inference_time = time.time() - start_inference
        logger.info(f"Inference completed in {inference_time:.2f}s")

        # Parse results
        detections = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # Get coordinates (normalized 0-1)
                    x1, y1, x2, y2 = box.xyxyn[0].tolist()
                    confidence = float(box.conf[0])
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    
                    # Convert to center format
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1
                    
                    detections.append({
                        "className": class_name,
                        "confidence": round(confidence, 3),
                        "boundingBox": {
                            "x": round(center_x - width/2, 4),
                            "y": round(center_y - height/2, 4),
                            "width": round(width, 4),
                            "height": round(height, 4)
                        }
                    })

        logger.info(f"Found {len(detections)} detections")
        
        return jsonify({
            "success": True,
            "detections": detections,
            "count": len(detections),
            "inference_time": round(inference_time, 2),
            "image_size": image.size
        })

    except Exception as e:
        logger.error(f"Detection failed: {e}", exc_info=True)
        return jsonify({
            "success": False,
            "error": f"Detection failed: {str(e)}"
        }), 500

@app.route('/classes', methods=['GET'])
def get_classes():
    """Get available classes"""
    if not MODEL_LOADED:
        return jsonify({
            "error": "Model not loaded"
        }), 500
    
    return jsonify({
        "classes": list(model.names.values()),
        "class_mapping": model.names,
        "count": len(model.names)
    })

@app.route('/status', methods=['GET'])
def status():
    """Detailed status"""
    return jsonify({
        "server": {
            "uptime": time.time() - start_time,
            "python": sys.version.split()[0]
        },
        "model": {
            "loaded": MODEL_LOADED,
            "loading": MODEL_LOADING,
            "error": MODEL_ERROR,
            "classes": len(model.names) if MODEL_LOADED and model else 0
        },
        "config": {
            "model_url_set": bool(MODEL_URL),
            "backup_url_set": bool(BACKUP_MODEL_URL),
            "max_image_size": MAX_IMAGE_SIZE
        }
    })

@app.route('/test', methods=['GET'])
def quick_test():
    """Quick test"""
    if not MODEL_LOADED:
        return jsonify({
            "success": False,
            "error": "Model not loaded"
        }), 500
    
    try:
        test_img = Image.new('RGB', (160, 160), 'blue')
        results = model(test_img, verbose=False, imgsz=160, conf=0.8)
        
        count = 0
        for result in results:
            if result.boxes is not None:
                count += len(result.boxes)
        
        return jsonify({
            "success": True,
            "test_detections": count,
            "model_classes": len(model.names),
            "message": "Test successful!"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

# Error handlers
@app.errorhandler(413)
def file_too_large(e):
    return jsonify({
        "success": False,
        "error": "File too large (max 5MB)"
    }), 413

@app.before_request
def limit_content_length():
    if request.content_length and request.content_length > MAX_CONTENT_LENGTH:
        return jsonify({
            "success": False,
            "error": "File too large (max 5MB)"
        }), 413

if __name__ == '__main__':
    # Start model loading immediately
    logger.info("🚀 Starting Railway app...")
    logger.info(f"Model URL configured: {bool(MODEL_URL)}")
    logger.info(f"Backup URL configured: {bool(BACKUP_MODEL_URL)}")
    
    if not MODEL_LOADED:
        threading.Thread(target=load_model, daemon=True).start()
    
    port = int(os.environ.get('PORT', 5000))
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        threaded=True
    )