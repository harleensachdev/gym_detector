from flask import Flask, request, jsonify
import cv2
import numpy as np
from ultralytics import YOLO
import base64
from io import BytesIO
from PIL import Image
import os

app = Flask(__name__)

# Initialize model (download your trained model to this directory)
MODEL_PATH = "best_model.pt"  # Your trained model file
model = None

def load_model():
    global model
    try:
        if os.path.exists(MODEL_PATH):
            model = YOLO(MODEL_PATH)
            print(f"Model loaded successfully from {MODEL_PATH}")
        else:
            print(f"Model file {MODEL_PATH} not found. Using YOLOv8n as fallback.")
            model = YOLO('yolov8n.pt')  # Fallback to pretrained model
    except Exception as e:
        print(f"Error loading model: {e}")
        model = YOLO('yolov8n.pt')  # Fallback

# Load model on startup
load_model()

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "model_path": MODEL_PATH
    })

@app.route('/detect', methods=['POST'])
def detect_objects():
    """Main detection endpoint"""
    try:
        # Check if model is loaded
        if model is None:
            return jsonify({"error": "Model not loaded"}), 500

        # Get image data from request
        if 'image' in request.files:
            # Handle file upload
            file = request.files['image']
            image_data = file.read()
        else:
            # Handle raw binary data
            image_data = request.get_data()

        if not image_data:
            return jsonify({"error": "No image data provided"}), 400

        # Convert bytes to PIL Image
        image = Image.open(BytesIO(image_data))
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Convert PIL to OpenCV format (numpy array)
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

        # Run inference
        results = model(opencv_image)

        # Parse results
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Get box coordinates (normalized 0-1)
                    x1, y1, x2, y2 = box.xyxyn[0].tolist()
                    
                    # Get confidence
                    confidence = float(box.conf[0])
                    
                    # Get class
                    class_id = int(box.cls[0])
                    class_name = model.names[class_id]
                    
                    # Convert to center format for consistency with your Swift code
                    center_x = (x1 + x2) / 2
                    center_y = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1
                    
                    detection = {
                        "className": class_name,
                        "confidence": confidence,
                        "boundingBox": {
                            "x": center_x - width/2,  # top-left x
                            "y": center_y - height/2, # top-left y
                            "width": width,
                            "height": height
                        }
                    }
                    
                    detections.append(detection)

        # Return results
        return jsonify({
            "success": True,
            "detections": detections,
            "count": len(detections)
        })

    except Exception as e:
        print(f"Detection error: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/classes', methods=['GET'])
def get_classes():
    """Get available classes"""
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    return jsonify({
        "classes": list(model.names.values()),
        "num_classes": len(model.names)
    })

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)