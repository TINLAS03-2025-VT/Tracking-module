import time
import cv2
from flask import Flask, Response

app = Flask(__name__)
output_frame = None

@app.route('/')
def index():
    return """
    <html>
    <head><title>AprilTag Stream</title></head>
    <body style="margin:0; background:#111; display:flex; justify-content:center; align-items:center; height:100vh;">
        <img src="/video_feed" style="max-width:100%; max-height:100%; object-fit:contain;">
    </body>
    </html>
    """

def generate_frames():
    global output_frame
    while True:
        if output_frame is None:
            time.sleep(0.03)
            continue

        # Downscale the frame for the web view
        downscaled = cv2.resize(output_frame, (640, 360), interpolation=cv2.INTER_AREA)

        # Encode as JPEG
        ret, buffer = cv2.imencode('.jpg', downscaled, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

def update_web_frame(color_frame):
    """
    Accepts an annotated frame, strips background color down to
    grayscale, but returns it as BGR format so existing color drawings
    pop brightly over a monochrome canvas.
    """
    global output_frame
    if color_frame is not None:
        gray_base = cv2.cvtColor(color_frame, cv2.COLOR_BGR2GRAY)
        output_frame = cv2.cvtColor(gray_base, cv2.COLOR_GRAY2BGR)

def run_flask(port=5000):
    app.run(host='0.0.0.0', port=port, threaded=True, use_reloader=False)