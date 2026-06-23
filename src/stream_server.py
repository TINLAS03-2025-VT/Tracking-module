import asyncio
import threading
from queue import Empty

import cv2

from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
import uvicorn

app = FastAPI(title="AprilTag Stream")

# Global reference within this child process space
_worker_queue = None
_latest_jpeg = None
_latest_lock = threading.Lock()

def jpeg_worker():
	global _worker_queue, _latest_jpeg

	while True:
		if _worker_queue is None:
			continue

		try:
			# Drain queue so we always keep only the newest frame
			frame = _worker_queue.get(timeout=1)

			while True:
				try:
					frame = _worker_queue.get_nowait()
				except Empty:
					break

			success, encoded_image = cv2.imencode(
				".jpg",
				frame,
				[int(cv2.IMWRITE_JPEG_QUALITY), 70],
			)

			if success:
				jpeg = encoded_image.tobytes()
				with _latest_lock:
					_latest_jpeg = jpeg

		except Empty:
			pass
		except Exception as e:
			print(f"JPEG worker error: {e}")


async def frame_generator():
	last_frame = None

	while True:
		with _latest_lock:
			frame = _latest_jpeg

		if frame is not None and frame is not last_frame:
			last_frame = frame
			yield (
				b"--frame\r\n"
				b"Content-Type: image/jpeg\r\n"
				b"Cache-Control: no-cache\r\n\r\n" +
				frame +
				b"\r\n"
			)

		await asyncio.sleep(0.01667)


@app.get("/stream")
async def video_stream():
	return StreamingResponse(
		frame_generator(),
		media_type="multipart/x-mixed-replace; boundary=frame",
		headers={
			"Cache-Control": "no-cache, no-store, must-revalidate",
			"Pragma": "no-cache",
			"Expires": "0",
		},
	)


@app.get("/")
async def index():
	html_content = """
	<html>
		<head>
			<title>AprilTag Live Monitor</title>
			<style>
				body {
					margin: 0;
					background: #111;
					display: flex;
					justify-content: center;
					align-items: center;
					height: 100vh;
				}
				img {
					max-width: 100%;
					max-height: 100%;
					object-fit: contain;
				}
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

	thread = threading.Thread(target=jpeg_worker, daemon=True)
	thread.start()

	uvicorn.run(app, host=host, port=port, log_level="info", loop="asyncio", workers=1)
