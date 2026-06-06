from flask import Flask, render_template, Response
import cv2
import numpy as np

app = Flask(__name__)

# Configured to your active IP Webcam mobile node
IP_WEBCAM_URL = "http://10.82.81.51:8080/video"
camera = cv2.VideoCapture(IP_WEBCAM_URL)

def generate_frames():
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Optimize frame geometry dimensions for fast vector mapping
            frame = cv2.resize(frame, (640, 480))
            
            # --- COMPUTER VISION & HEATMAP TRANSFORMS ---
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray_frame, (25, 25), 0)
            
            # Convert single channel grayscale arrays to COLORMAP_JET array matrices
            heatmap_layer = cv2.applyColorMap(blurred, cv2.COLORMAP_JET)
            
            # Fuse spatial arrays together to overlay heat maps onto live stream
            processed_output = cv2.addWeighted(frame, 0.6, heatmap_layer, 0.4, 0)
            
            # Compress mathematical canvas matrices into a standard JPEG byte block
            ret, encoding_buffer = cv2.imencode('.jpg', processed_output)
            frame_bytes = encoding_buffer.tobytes()
            
            # Yield frame stream chunks to browser pipeline sequentially
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
