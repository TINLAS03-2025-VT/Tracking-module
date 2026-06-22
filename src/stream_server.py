import asyncio
from queue import Empty
import threading
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
import uvicorn
import cv2

app = FastAPI(title="Optimized AprilTag Stream")

# Global reference within this child process space
_worker_queue = None

async def frame_generator():
	global _worker_queue
	last_sent_frame = None
	while True:
		if _worker_queue is not None:
			try:
				frame = _worker_queue.get_nowait()
				success, encoded_image = cv2.imencode(
					'.jpg',
					frame,
					[int(cv2.IMWRITE_JPEG_QUALITY), 75]
				)
				if success:
					yield (b'--frame\r\n'
							b'Content-Type: image/jpeg\r\n\r\n' + encoded_image.tobytes() + b'\r\n')
			except Empty:
				pass

		# High frequency sleep (~60Hz) ensures the socket stays awake
		# and doesn't trigger NS_ERROR_NET_EMPTY_RESPONSE
		await asyncio.sleep(0.016)

@app.get("/stream")
async def video_stream():
	return StreamingResponse(
		frame_generator(),
		media_type="multipart/x-mixed-replace; boundary=frame"
	)

@app.get("/")
async def index():
	html_content = """
	<html>
		<head>
			<title>AprilTag Live Monitor</title>
			<style>
				body { margin: 0; background: #111; display: flex; justify-content: center; align-items: center; height: 100vh; }
				img { max-width: 100%; max-height: 100%; object-fit: contain; }
			</style>
		</head>
		<body>
			<img src="/stream" />
		</body>
	</html>
	"""
	return HTMLResponse(content=html_content, status_code=200)

def start_stream_server(queue, host: str = "0.0.0.0", port: int = 8080):
	global _worker_queue
	_worker_queue = queue
	uvicorn.run(app, host=host, port=port, log_level="info", loop="asyncio", workers=1)