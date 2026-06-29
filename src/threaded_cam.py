import threading
import cv2

class ThreadedCamera:
	def __init__(self, camera_index, width=1280, height=720):
		self.cap = cv2.VideoCapture(camera_index)
		self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
		self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
		self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

		self.ret, self.frame = False, None
		self.started = False
		self.read_lock = threading.Lock()

	def start(self):
		if self.started:
			return self
		self.started = True
		self.thread = threading.Thread(target=self._update, args=(), daemon=True)
		self.thread.start()
		return self

	def _update(self):
		while self.started:
			ret, frame = self.cap.read()
			if not ret:
				continue
			with self.read_lock:
				self.ret = ret
				self.frame = frame

	def read(self):
		with self.read_lock:
			# Return a shallow copy so the thread can safely overwrite the buffer
			return self.ret, self.frame.copy() if self.frame is not None else None

	def release(self):
		self.started = False
		if hasattr(self, 'thread'):
			self.thread.join(timeout=1.0)
		self.cap.release()
