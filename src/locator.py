import numpy as np
import apriltag
from scipy.spatial.transform import Rotation, Slerp

class Locator:
	def __init__(
		self,
		cam_cal = {"fx": 500, "fy": 500, "cx": 320, "cy": 240},
		tag_size = 0.15,
		alpha_lateral = 0.15,
		alpha_forward = 0.25,
		alpha_rotation = 0.20,
		pylon_tags_ids = (0, 1)
	):
		self.detector = apriltag.apriltag(
			family='tagStandard41h12',   # Tag family
			threads=10,                  # Number of threads
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
		self.ALPHA_ROTATION = alpha_rotation
		self.PYLON_TAG_IDS = pylon_tags_ids

		self.poses = {}

	def detect(self, grayscale_frame):
		return self.detector.detect(grayscale_frame)

	def get_poses(self, detections):
		current_frame_ids = set()

		for detection in detections:
			tag_id = detection["id"]
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

			try:
				rot_prev = Rotation.from_matrix(prev_R)
				rot_raw = Rotation.from_matrix(raw_R)

				# Create a keyframe interpolation structure for times [0, 1]
				times = [0.0, 1.0]
				quats = Rotation.from_quat([rot_prev.as_quat(), rot_raw.as_quat()])
				slerp = Slerp(times, quats)

				# Interpolate based on your alpha_rotation value
				stable_rotation = slerp([self.ALPHA_ROTATION])
				stable_R = stable_rotation.as_matrix()[0]
			except Exception:
				# Fallback to raw if matrix noise forces mathematical singularities
				stable_R = raw_R

			self.poses[tag_id] = {"t": stable_t, "R": stable_R}

		self.poses = {tag_id: pose for tag_id, pose in self.poses.items() if tag_id in current_frame_ids}

		return self.poses
