import time
import cv2
import numpy as np
import apriltag

class Locator:
	def __init__(
		self,
		cam_cal = {"fx": 500, "fy": 500, "cx": 320, "cy": 240},
		tag_size = 0.15,
		alpha_lateral = 0.15,
		alpha_forward = 0.25,
		pylon_tags_ids = (0, 1),
		roi_size = 200,
		full_scan_interval = 0.5
	):
		self.detector = apriltag.apriltag(
			family='tagStandard41h12',   # Tag family
			threads=6,                  # Number of threads
			maxhamming=1,                # Maximum hamming distance for error correction
			decimate=1.0,                # Image downsampling factor
			blur=0.0,                    # Gaussian blur sigma
			refine_edges=True,           # Refine quad edges
			debug=False                  # Debug mode
		)

		self.CAM_CAL = cam_cal
		self.TAG_SIZE = tag_size

		self.ALPHA_LATERAL = alpha_lateral
		self.ALPHA_FORWARD = alpha_forward
		self.PYLON_TAG_IDS = pylon_tags_ids

		self.ROI_SIZE = roi_size
		self.FULL_SCAN_INTERVAL = full_scan_interval
		self.last_full_scan_time = 0.0
		self.last_pixel_centers = {}

		self.poses = {}

	def detect(self, grayscale_frame, wanted_tag_ids=None):
		detections = self.detector.detect(grayscale_frame)

		if wanted_tag_ids is None:
			return detections

		filtered_detections = [
			det for det in detections if det["id"] in wanted_tag_ids
		]

		return filtered_detections

	def get_poses(self, detections, wanted_tag_ids=None):
		current_frame_ids = set()

		if wanted_tag_ids is None:
			return {}

		for detection in detections:
			tag_id = detection["id"]
			if tag_id not in wanted_tag_ids:
				continue

			current_frame_ids.add(tag_id)

			raw_pose = self.detector.estimate_tag_pose(
				detection,
				self.TAG_SIZE,
				self.CAM_CAL["fx"], self.CAM_CAL["fy"],
				self.CAM_CAL["cx"], self.CAM_CAL["cy"]
			)

			raw_t = np.array(raw_pose["t"]).flatten()
			raw_R = np.array(raw_pose["R"])

			if tag_id not in self.poses:
				self.poses[tag_id] = {"t": raw_t, "R": raw_R}
				continue

			prev_pose = self.poses[tag_id]
			prev_t = prev_pose["t"]
			prev_R = prev_pose["R"]

			alphas = np.array([self.ALPHA_LATERAL, self.ALPHA_LATERAL, self.ALPHA_FORWARD])
			stable_t = (alphas * raw_t) + ((1.0 - alphas) * prev_t)

			self.poses[tag_id] = {"t": stable_t, "R": raw_R}

		self.poses = {tag_id: pose for tag_id, pose in self.poses.items() if tag_id in current_frame_ids}

		return self.poses
