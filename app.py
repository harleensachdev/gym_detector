from flask import Flask, request, jsonify
import os
import sys
import time
import threading
import gc
import psutil
from io import BytesIO
from PIL import Image
import logging
import requests
import hashlib
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Track startup time for health checks
start_time = time.time()

app = Flask(__name__)

# Global model variables
model = None
MODEL_LOADED = False
MODEL_LOADING = False
MODEL_LOAD_ERROR = None

# Configuration
class Config:
    # Railway optimized settings
    MAX_CONTENT_LENGTH = 3 * 1024 * 1024  # 3MB max file size
    MODEL_CACHE_DIR = "/tmp/models"  # Railway temp storage
    IMAGE_MAX_SIZE = 512  # Reduced for Railway memory limits
    INFERENCE_TIMEOUT = 25  # Timeout for inference (Railway has 30s request timeout)
    
    # Model download URLs - Set these as Railway environment variables
    MODEL_URLS = {
        "primary": os.getenv("PRIMARY_MODEL_URL", ""),  # Your custom model URL
        "backup": os.getenv("BACKUP_MODEL_URL", ""),    # Backup model URL
        "fallback": "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt"
    }
    
    # Model file names
    MODEL_FILES = {
        "primary": "custom_model.pt",
        "backup": "backup_model.pt", 
        "fallback": "yolov8n.pt"
    }

app.config.update(vars(Config))

def ensure_model_dir():
    """Create model cache directory"""
    os.makedirs(Config.MODEL_CACHE_DIR, exist_ok=True)
    logger.info(f"Model cache directory: {Config.MODEL_CACHE_DIR}")

def download_model(url, filename, max_retries=3):
    """Download model with retry logic and progress tracking"""
    if not url:
        logger.warning(f"No URL provided for {filename}")
        return False
    
    filepath = os.path.join(Config.MODEL_CACHE_DIR, filename)
    
    # Check if file already exists and is valid
    if os.path.exists(filepath):
        file_size = os.path.getsize(filepath)
        if file_size > 1024 * 1024:  # At least 1MB
            logger.info(f"Model {filename} already exists ({file_size} bytes)")
            return True
    
    logger.info(f"Downloading model from {url}")
    
    for attempt in range(max_retries):
        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            with open(filepath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        
                        # Log progress every 10MB
                        if downloaded_size % (10 * 1024 * 1024) == 0:
                            progress = (downloaded_size / total_size) * 100 if total_size > 0 else 0
                            logger.info(f"Download progress: {progress:.1f}% ({downloaded_size / 1024 / 1024:.1f}MB)")
            
            file_size = os.path.getsize(filepath)
            logger.info(f"Successfully downloaded {filename} ({file_size} bytes)")
            return True
            
        except Exception as e:
            logger.error(f"Download attempt {attempt + 1} failed: {e}")
            if os.path.exists(filepath):
                os.remove(filepath)
            
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Exponential backoff
    
    logger.error(f"Failed to download {filename} after {max_retries} attempts")
    return False

def load_model_with_fallback():
    """Load model with fallback strategy"""
    global model, MODEL_LOADED, MODEL_LOADING, MODEL_LOAD_ERROR
    
    if MODEL_LOADED or MODEL_LOADING:
        return
    
    MODEL_LOADING = True
    MODEL_LOAD_ERROR = None
    logger.info("Starting model loading process...")
    
    try:
        from ultralytics import YOLO
        logger.info("Successfully imported ultralytics")
        
        ensure_model_dir()
        
        # Try loading models in order of preference
        model_loaded = False
        
        # 1. Try primary custom model
        if Config.MODEL_URLS["primary"]:
            logger.info("Attempting to load primary custom model...")
            if download_model(Config.MODEL_URLS["primary"], Config.MODEL_FILES["primary"]):
                try:
                    model_path = os.path.join(Config.MODEL_CACHE_DIR, Config.MODEL_FILES["primary"])
                    model = YOLO(model_path)
                    logger.info(f"✅ Primary custom model loaded successfully with {len(model.names)} classes")
                    model_loaded = True
                except Exception as e:
                    logger.error(f"Failed to load primary model: {e}")
        
        # 2. Try backup model
        if not model_loaded and Config.MODEL_URLS["backup"]:
            logger.info("Attempting to load backup model...")
            if download_model(Config.MODEL_URLS["backup"], Config.MODEL_FILES["backup"]):
                try:
                    model_path = os.path.join(Config.MODEL_CACHE_DIR, Config.MODEL_FILES["backup"])
                    model = YOLO(model_path)
                    logger.info(f"✅ Backup model loaded successfully with {len(model.names)} classes")
                    model_loaded = True
                except Exception as e:
                    logger.error(f"Failed to load backup model: {e}")
        
        # 3. Try local files (if any exist)
        if not model_loaded:
            local_model_files = [
                "best_model.pt", "model_epoch_85.pt", "best.pt", "last.pt", "yolov8n.pt"
            ]
            
            for model_file in local_model_files:
                if os.path.exists(model_file):
                    try:
                        logger.info(f"Found local model: {model_file}")
                        model = YOLO(model_file)
                        logger.info(f"✅ Local model {model_file} loaded with {len(model.names)} classes")
                        model_loaded = True
                        break
                    except Exception as e:
                        logger.error(f"Failed to load local model {model_file}: {e}")
        
        # 4. Fallback to YOLOv8n
        if not model_loaded:
            logger.info("Loading fallback YOLOv8n model...")
            if download_model(Config.MODEL_URLS["fallback"], Config.MODEL_FILES["fallback"]):
                try:
                    model_path = os.path.join(Config.MODEL_CACHE_DIR, Config.MODEL_FILES["fallback"])
                    model = YOLO(model_path)
                    logger.info("✅ Fallback YOLOv8n model loaded successfully")
                    model_loaded = True
                except Exception as e:
                    logger.error(f"Failed to load fallback model: {e}")
        
        if not model_loaded:
            # Last resort - try downloading YOLOv8n directly
            try:
                logger.info("Last resort: Loading YOLOv8n directly...")
                model = YOLO('yolov8n.pt')  # This will auto-download
                logger.info("✅ YOLOv8n loaded directly")
                model_loaded = True
            except Exception as e:
                logger.error(f"Even YOLOv8n failed to load: {e}")
                MODEL_LOAD_ERROR = f"All model loading attempts failed: {e}"
                return
        
        if model_loaded:
            # Warm up the model with a small test image
            logger.info("Warming up model...")
            try:
                dummy_img = Image.new('RGB', (320, 320), color='red')
                _ = model(dummy_img, verbose=False, imgsz=320, conf=0.8, max_det=1)
                logger.info("✅ Model warmed up successfully")
                MODEL_LOADED = True
            except Exception as e:
                logger.error(f"Model warmup failed: {e}")
                MODEL_LOAD_ERROR = f"Model warmup failed: {e}"
        
    except ImportError as e:
        error_msg = f"Failed to import required packages: {e}"
        logger.error(error_msg)
        MODEL_LOAD_ERROR = error_msg
    except Exception as e:
        error_msg = f"Unexpected error during model loading: {e}"
        logger.error(error_msg, exc_info=True)
        MODEL_LOAD_ERROR = error_msg
    finally:
        MODEL_LOADING = False
        # Force garbage collection after model loading
        gc.collect()

def optimize_image(image, max_size=None):
    """Optimize image for Railway's memory constraints"""
    if max_size is None:
        max_size = Config.IMAGE_MAX_SIZE
    
    width, height = image.size
    original_size = width * height
    
    if max(width, height) > max_size:
        if width > height:
            new_width = max_size
            new_height = int((height * max_size) / width)
        else:
            new_height = max_size
            new_width = int((width * max_size) / height)
        
        logger.info(f"Resizing image: {width}x{height} -> {new_width}x{new_height}")
        image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    
    # Convert to RGB if necessary
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    return image

def cleanup_memory():
    """Aggressive memory cleanup for Railway"""
    gc.collect()
    if hasattr(gc, 'set_threshold'):
        gc.set_threshold(700, 10, 10)  # More aggressive GC

def get_memory_usage():
    """Get current memory usage"""
    try:
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()
        return {
            "rss": memory_info.rss / 1024 / 1024,  # MB
            "vms": memory_info.vms / 1024 / 1024,  # MB
            "percent": memory_percent
        }
    except:
        return {"error": "Could not get memory info"}

# Configure Flask
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH

@app.before_request
def limit_content_length():
    """Reject oversized requests early"""
    if request.content_length and request.content_length > Config.MAX_CONTENT_LENGTH:
        return jsonify({
            "success": False,
            "error": f"Image too large. Please use a smaller image (under {Config.MAX_CONTENT_LENGTH // 1024 // 1024}MB)."
        }), 413

@app.route('/', methods=['GET'])
def health_check():
    """Comprehensive health check"""
    memory_info = get_memory_usage()
    uptime = time.time() - start_time
    
    return jsonify({
        "status": "healthy" if MODEL_LOADED else ("loading" if MODEL_LOADING else "error"),
        "model_loaded": MODEL_LOADED,
        "model_loading": MODEL_LOADING,
        "model_error": MODEL_LOAD_ERROR,
        "uptime_seconds": round(uptime, 2),
        "memory_usage_mb": round(memory_info.get("rss", 0), 2),
        "memory_percent": round(memory_info.get("percent", 0), 2),
        "available_classes": list(model.names.values()) if MODEL_LOADED and model else [],
        "class_count": len(model.names) if MODEL_LOADED and model else 0,
        "python_version": sys.version.split()[0],
        "message": "YOLO Custom Model Detection API - Railway Optimized"
    })

@app.route('/start-loading', methods=['POST'])
def start_model_loading():
    """Manually trigger model loading (useful for Railway deployments)"""
    global MODEL_LOADING
    
    if MODEL_LOADED:
        return jsonify({
            "success": True,
            "message": "Model already loaded",
            "classes": list(model.names.values())
        })
    
    if MODEL_LOADING:
        return jsonify({
            "success": False,
            "message": "Model is already loading, please wait"
        }), 202
    
    # Start loading in background thread
    threading.Thread(target=load_model_with_fallback, daemon=True).start()
    
    return jsonify({
        "success": True,
        "message": "Model loading started in background"
    }), 202

@app.route('/detect', methods=['POST'])
def detect_objects():
    """Main detection endpoint - optimized for large custom models on Railway"""
    
    # Auto-start model loading if not started
    if not MODEL_LOADED and not MODEL_LOADING:
        logger.info("Model not loaded, starting loading process...")
        threading.Thread(target=load_model_with_fallback, daemon=True).start()
    
    # Check model status
    if MODEL_LOADING:
        return jsonify({
            "success": False,
            "error": "Model is still loading, please wait and try again",
            "retry_after": 15,
            "suggestion": "Call /start-loading first, then wait 30-60 seconds before detecting"
        }), 503
    
    if not MODEL_LOADED:
        error_msg = MODEL_LOAD_ERROR if MODEL_LOAD_ERROR else "Model failed to load"
        return jsonify({
            "success": False,
            "error": error_msg,
            "suggestion": "Check server logs or try calling /start-loading"
        }), 500
    
    start_time_detect = time.time()
    
    try:
        # Clean up memory before processing
        cleanup_memory()
        
        # Get image data
        if 'image' in request.files:
            file = request.files['image']
            image_data = file.read()
            logger.info(f"File upload received: {len(image_data)} bytes")
        else:
            image_data = request.get_data()
            logger.info(f"Raw data received: {len(image_data)} bytes")

        if not image_data or len(image_data) < 100:
            return jsonify({
                "success": False,
                "error": "No valid image data received"
            }), 400

        # Convert to PIL Image
        try:
            image = Image.open(BytesIO(image_data))
            original_size = image.size
            logger.info(f"Original image: {original_size}")
            
            # Optimize image for Railway memory constraints
            image = optimize_image(image, max_size=Config.IMAGE_MAX_SIZE)
            optimized_size = image.size
            logger.info(f"Optimized image: {optimized_size}")
            
        except Exception as e:
            logger.error(f"Image processing failed: {e}")
            return jsonify({
                "success": False,
                "error": f"Invalid image format: {str(e)}"
            }), 400

        # Run inference with timeout protection
        logger.info("Starting inference...")
        inference_start = time.time()
        
        try:
            # Use conservative settings for Railway
            results = model(
                image, 
                verbose=False,
                imgsz=min(640, max(optimized_size)),  # Dynamic image size
                conf=0.25,      # Reasonable confidence threshold
                iou=0.45,       # Standard IoU threshold
                max_det=100,    # Reasonable max detections
                device='cpu',   # Force CPU (Railway doesn't have GPU)
                half=False,     # Disable half precision
                augment=False,  # Disable augmentation for speed
                agnostic_nms=False
            )
            
            inference_time = time.time() - inference_start
            logger.info(f"Inference completed in {inference_time:.2f}s")
            
            # Check if inference took too long
            if inference_time > Config.INFERENCE_TIMEOUT:
                logger.warning(f"Inference took {inference_time:.2f}s (longer than {Config.INFERENCE_TIMEOUT}s)")
            
        except Exception as e:
            logger.error(f"Model inference failed: {e}")
            cleanup_memory()
            return jsonify({
                "success": False,
                "error": f"Model inference failed: {str(e)}",
                "suggestion": "Try a smaller image or check model compatibility"
            }), 500

        # Parse results efficiently
        detections = []
        try:
            for result in results:
                if result.boxes is not None and len(result.boxes) > 0:
                    logger.info(f"Processing {len(result.boxes)} detections")
                    
                    for box in result.boxes:
                        try:
                            # Get normalized coordinates (0-1)
                            x1, y1, x2, y2 = box.xyxyn[0].cpu().numpy().tolist()
                            confidence = float(box.conf[0].cpu().numpy())
                            class_id = int(box.cls[0].cpu().numpy())
                            
                            # Get class name safely
                            class_name = model.names.get(class_id, f"class_{class_id}")
                            
                            # Convert to center format for consistency
                            center_x = (x1 + x2) / 2
                            center_y = (y1 + y2) / 2
                            width = x2 - x1
                            height = y2 - y1
                            
                            detection = {
                                "className": class_name,
                                "confidence": round(confidence, 3),
                                "boundingBox": {
                                    "x": round(center_x - width/2, 4),
                                    "y": round(center_y - height/2, 4),
                                    "width": round(width, 4),
                                    "height": round(height, 4)
                                }
                            }
                            
                            detections.append(detection)
                            logger.info(f"Detection: {class_name} ({confidence:.3f})")
                            
                        except Exception as e:
                            logger.error(f"Error processing detection: {e}")
                            continue
                
        except Exception as e:
            logger.error(f"Results parsing failed: {e}")
        
        # Clean up memory after processing
        cleanup_memory()
        
        total_time = time.time() - start_time_detect
        memory_after = get_memory_usage()
        
        logger.info(f"Detection completed: {len(detections)} objects in {total_time:.2f}s")
        
        return jsonify({
            "success": True,
            "detections": detections,
            "count": len(detections),
            "processing_time": round(total_time, 2),
            "inference_time": round(inference_time, 2),
            "memory_usage_mb": round(memory_after.get("rss", 0), 2),
            "image_info": {
                "original_size": original_size,
                "processed_size": optimized_size
            },
            "model_info": {
                "classes": len(model.names),
                "model_type": "custom" if "custom" in str(model.ckpt_path) else "standard"
            }
        })

    except Exception as e:
        logger.error(f"Detection error: {str(e)}", exc_info=True)
        cleanup_memory()
        return jsonify({
            "success": False,
            "error": f"Processing failed: {str(e)}",
            "suggestion": "Try a smaller image, check image format, or contact support"
        }), 500

@app.route('/classes', methods=['GET'])
def get_classes():
    """Get available detection classes"""
    if MODEL_LOADING:
        return jsonify({
            "status": "loading",
            "message": "Model is still loading"
        }), 503
    
    if not MODEL_LOADED or model is None:
        return jsonify({
            "error": "Model not loaded",
            "error_details": MODEL_LOAD_ERROR
        }), 500
    
    return jsonify({
        "classes": list(model.names.values()),
        "class_mapping": model.names,
        "num_classes": len(model.names),
        "model_loaded": True
    })

@app.route('/status', methods=['GET'])
def get_status():
    """Detailed status endpoint for monitoring"""
    memory_info = get_memory_usage()
    uptime = time.time() - start_time
    
    status = {
        "server": {
            "status": "running",
            "uptime_seconds": round(uptime, 2),
            "memory_usage": memory_info,
            "python_version": sys.version.split()[0]
        },
        "model": {
            "loaded": MODEL_LOADED,
            "loading": MODEL_LOADING,
            "error": MODEL_LOAD_ERROR,
            "classes": len(model.names) if MODEL_LOADED and model else 0,
            "available_classes": list(model.names.values()) if MODEL_LOADED and model else []
        },
        "config": {
            "max_image_size": Config.IMAGE_MAX_SIZE,
            "max_upload_size_mb": Config.MAX_CONTENT_LENGTH // 1024 // 1024,
            "inference_timeout": Config.INFERENCE_TIMEOUT
        }
    }
    
    return jsonify(status)

@app.route('/quick-test', methods=['GET'])
def quick_test():
    """Fast test endpoint that doesn't load the full model"""
    if MODEL_LOADING:
        return jsonify({
            "success": False,
            "status": "loading",
            "message": "Model is still loading"
        }), 503
    
    if not MODEL_LOADED or model is None:
        return jsonify({
            "success": False,
            "status": "error",
            "message": "Model not loaded",
            "error": MODEL_LOAD_ERROR
        }), 500
    
    try:
        # Create tiny test image
        test_image = Image.new('RGB', (160, 160), color='blue')
        
        # Run ultra-fast detection
        start_time_test = time.time()
        results = model(test_image, verbose=False, imgsz=160, conf=0.9, max_det=1)
        test_time = time.time() - start_time_test
        
        detection_count = 0
        for result in results:
            if result.boxes is not None:
                detection_count += len(result.boxes)
        
        return jsonify({
            "success": True,
            "message": "Quick test completed successfully",
            "test_time": round(test_time, 3),
            "detections_found": detection_count,
            "server_status": "responsive",
            "model_classes": len(model.names)
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Quick test failed: {str(e)}",
            "server_status": "error"
        }), 500

# Error handlers
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        "success": False,
        "error": "File too large",
        "max_size_mb": Config.MAX_CONTENT_LENGTH // 1024 // 1024
    }), 413

@app.errorhandler(500)
def internal_server_error(error):
    return jsonify({
        "success": False,
        "error": "Internal server error",
        "suggestion": "Please try again or contact support"
    }), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    
    # Start model loading in background immediately
    if not MODEL_LOADED and not MODEL_LOADING:
        logger.info("Starting background model loading...")
        threading.Thread(target=load_model_with_fallback, daemon=True).start()
    
    logger.info(f"Starting Flask app on port {port}")
    
    # Production-optimized settings for Railway
    app.run(
        host='0.0.0.0', 
        port=port, 
        debug=False,
        threaded=True,
        use_reloader=False
    )