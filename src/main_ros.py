import argparse
import json
import math
import os
import time
from threading import Thread # Added for web server
from typing import Dict, List, Tuple

import apriltag
import cv2
import numpy as np

# Flask dependencies
from flask import Flask, Response

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose, PoseStamped

import defines

# --- Flask App Setup ---
app = Flask(__name__)
output_frame = None

@app.route('/')
def index():
	# A tiny HTML wrapper to display just the image
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

		# Downscale the frame for the web view (e.g., to 640x480 or half size)
		downscaled = cv2.resize(output_frame, (1280, 720), interpolation=cv2.INTER_AREA)

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

def run_flask():
	# Runs the web server on port 5000, visible externally
	app.run(host='0.0.0.0', port=5000, threaded=True, use_reloader=False)

def parse_args():
	valid_profiles = [
		attr for attr in dir(defines)
		if not attr.startswith("__") and not hasattr(getattr(defines, attr), "__call__")
	]
	profiles_str = ", ".join(valid_profiles) if valid_profiles else "none found"

	parser = argparse.ArgumentParser(description="Headless AprilTag tracker")
	parser.add_argument("-c", "--camera-index", type=int, default=int(os.getenv("CAMERA_INDEX", "0")), help="index of the camera to use (default: 0)")
	parser.add_argument("-s", "--show", action="store_true", help="show OpenCV debug window")
	parser.add_argument("-t", "--tag-size", type=float, default=float(os.getenv("TAG_SIZE", "0.025")), help="inner dimensions of the tags in meters (default: 0.025)")
	parser.add_argument("-p", "--camera-profile", default=os.getenv("CAMERA_PROFILE", "microsoft_cam"), choices=valid_profiles, help=f"camera calibration profile from defines.py. Available options: {profiles_str}")
	parser.add_argument("-x", "--scale-x", type=float, default=float(os.getenv("SCALE_X", "200")), help="x-axis scale to map the locations to (default: 200.0)")
	parser.add_argument("-y", "--scale-y", type=float, default=float(os.getenv("SCALE_Y", "200")), help="y-axis scale to map the locations to (default: 200.0)")
	parser.add_argument("-0", "--reference-tag-0", type=int, default=int(os.getenv("REFERENCE_TAG_0", "0")), help="tag ID for reference tag at location {x: 0, y: 0} (default: 0)")
	parser.add_argument("-1", "--reference-tag-1", type=int, default=int(os.getenv("REFERENCE_TAG_1", "1")), help="tag ID for reference tag at location {x: SCALE_X, y: SCALE_Y} (default: 1)")
	parser.add_argument("-a", "--alpha", type=float, default=float(os.getenv("TRACKER_ALPHA", "0.3")), help="weight alpha for exponential moving average (default: 0.3)")
	parser.add_argument("--target-tags", default=os.getenv("TARGET_TAGS", "2:robot_1,3:robot_2"), help="Mapping target tags to robot names")
	parser.add_argument("--output-topic", default=os.getenv("OUTPUT_TOPIC", "/robots/pos"), help="ROS 2 topic for aggregate poses")
	parser.add_argument("--publish-individual-poses", default=os.getenv("PUBLISH_INDIVIDUAL_POSES", "true").lower() == "true", action=argparse.BooleanOptionalAction, help="Publish independent PoseStamped topics for each robot")
	return parser.parse_args()

def parse_target_tags(raw: str) -> Dict[int, str]:
	result = {}
	for index, item in enumerate(raw.split(","), start=1):
		item = item.strip()
		if not item:
			continue
		if ":" in item:
			tag_str, robot_id = item.split(":", 1)
			result[int(tag_str.strip())] = robot_id.strip()
		else:
			result[int(item)] = f"robot_{index}"
	return result


def get_camera_calibration(profile_name):
	if not hasattr(defines, profile_name):
		raise ValueError(f"Unknown camera profile: {profile_name}")
	return getattr(defines, profile_name)

def make_4x4_matrix(R, t):
	T = np.eye(4, dtype=np.float32)
	T[0:3, 0:3] = R
	T[0:3, 3] = t.flatten()
	return T


def yaw_to_pose_orientation(pose: Pose, yaw_rad: float):
	half = yaw_rad / 2.0
	pose.orientation.x = 0.0
	pose.orientation.y = 0.0
	pose.orientation.z = math.sin(half)
	pose.orientation.w = math.cos(half)

class AprilTagRosTracker(Node):
	def __init__(self, args):
		super().__init__("apriltag_tracker")

		self.args = args
		self.target_tag_map = parse_target_tags(args.target_tags)

		self.pose_array_pub = self.create_publisher(PoseArray, args.output_topic, 10)
		self.individual_pubs = {}

		self.get_logger().info(f"Publishing aggregate poses to {args.output_topic}")
		self.get_logger().info(f"Target tag map: {self.target_tag_map}")

	def get_individual_pub(self, robot_id: str):
		if robot_id not in self.individual_pubs:
			topic = f"/{robot_id}/pose"
			self.individual_pubs[robot_id] = self.create_publisher(PoseStamped, topic, 10)
			self.get_logger().info(f"Publishing individual pose to {topic}")
		return self.individual_pubs[robot_id]

	def publish_poses(self, robots: List[Tuple[str, float, float, float]]):
		pose_array = PoseArray()
		pose_array.header.stamp = self.get_clock().now().to_msg()
		pose_array.header.frame_id = "map"

		for robot_id, x, y, theta in robots:
			pose = Pose()
			pose.position.x = float(x)
			pose.position.y = float(y)
			pose.position.z = 0.0
			yaw_to_pose_orientation(pose, theta)
			pose_array.poses.append(pose)

			if self.args.publish_individual_poses:
				stamped = PoseStamped()
				stamped.header = pose_array.header
				stamped.pose = pose
				self.get_individual_pub(robot_id).publish(stamped)

		self.pose_array_pub.publish(pose_array)


def main():
	global output_frame
	args = parse_args()
	cam_cal = get_camera_calibration(args.camera_profile)
	fx, fy, cx, cy = cam_cal["fx"], cam_cal["fy"], cam_cal["cx"], cam_cal["cy"]

	rclpy.init()
	node = AprilTagRosTracker(args)

	flask_thread = Thread(target=run_flask, daemon=True)
	flask_thread.start()

	detector = apriltag.apriltag(
		family="tagStandard41h12",
		threads=8,
		refine_edges=True,
		debug=False,
	)

	cap = cv2.VideoCapture(args.camera_index)
	if not cap.isOpened():
		node.get_logger().error(f"Could not open camera index {args.camera_index}")
		raise RuntimeError(f"Could not open camera index {args.camera_index}")

	tracked_robots = {}
	angle_buffers = {}

	map_calibrated = False
	R_ref0_to_stable = None
	scale_factor = 1.0
	theta_rad = 0.0

	smoothed_R_ref0 = None
	smoothed_scale_factor = 1.0
	smoothed_theta_rad = 0.0
	calibration_frames_tracked = 0
	CALIBRATION_LOCK_THRESHOLD = 50

	print(json.dumps({
		"event": "ros_tracker_started",
		"camera_index": args.camera_index,
		"camera_profile": args.camera_profile,
		"reference_tags": [args.reference_tag_0, args.reference_tag_1],
		"target_tag_map": node.target_tag_map,
		"scale": [args.scale_x, args.scale_y],
		"output_topic": args.output_topic,
		"publish_individual_poses": args.publish_individual_poses,
		"show": args.show,
	}), flush=True)

	try:
		while rclpy.ok():
			ret, frame = cap.read()
			if not ret:
				node.get_logger().warn("Camera frame dropped.")
				time.sleep(0.2)
				continue

			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			detections = detector.detect(gray)

			frame_poses = {}
			frame_centers = {}

			for det in detections:
				tag_id = int(det["id"])
				center = np.array(det["center"], dtype=np.float32)
				frame_centers[tag_id] = center

				pose = detector.estimate_tag_pose(det, args.tag_size, fx, fy, cx, cy)
				frame_poses[tag_id] = make_4x4_matrix(pose['R'], pose['t'])

				if args.show:
					corners = np.array(det["lb-rb-rt-lt"], dtype=np.int32)
					center_int = tuple(center.astype(int))
					cv2.polylines(frame, [corners], True, (0, 255, 0), 2)
					cv2.circle(frame, center_int, 5, (0, 0, 255), -1)
					cv2.putText(frame, str(tag_id), center_int, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

			if args.reference_tag_0 in frame_poses and args.reference_tag_1 in frame_poses:
				if calibration_frames_tracked < CALIBRATION_LOCK_THRESHOLD or CALIBRATION_LOCK_THRESHOLD == 0:
					T_cam_ref0 = frame_poses[args.reference_tag_0]
					T_cam_ref1 = frame_poses[args.reference_tag_1]
					T_ref0_cam = np.linalg.inv(T_cam_ref0)

					p1_in_ref0 = T_ref0_cam @ np.append(T_cam_ref1[0:3, 3], 1.0)
					v_ref0 = p1_in_ref0[0:3]
					d_phys = np.linalg.norm(v_ref0)

					if d_phys > 0.001:
						n_ref0 = np.array([0.0, 0.0, 1.0], dtype=np.float32)
						u_y_ref0= np.cross(n_ref0, v_ref0)
						u_y_norm = np.linalg.norm(u_y_ref0)

						if u_y_norm > 0.001:
							u_y_ref0 /= u_y_norm
							u_z_ref0 = np.cross(v_ref0, u_y_ref0)
							u_z_ref0 /= np.linalg.norm(u_z_ref0)
							u_x_ref0 = np.cross(u_y_ref0, u_z_ref0)
							u_x_ref0 /= np.linalg.norm(u_x_ref0)

							raw_R = np.stack([u_x_ref0, u_y_ref0, u_z_ref0], axis=1)
							raw_scale = math.hypot(args.scale_x, args.scale_y) / d_phys
							raw_theta = math.atan2(args.scale_y, args.scale_x)

							cal_alpha = 0.1

							if smoothed_R_ref0 is None:
								smoothed_R_ref0 = raw_R
								smoothed_scale_factor = raw_scale
								smoothed_theta_rad = raw_theta
							else:
								smoothed_R_ref0 = (cal_alpha * raw_R) + ((1.0 - cal_alpha) * smoothed_R_ref0)
								smoothed_scale_factor = (cal_alpha * raw_scale) + ((1.0 - cal_alpha) * smoothed_scale_factor)
								# Handle angle wrap-around safely
								smoothed_theta_rad = smoothed_theta_rad + cal_alpha * math.atan2(
									math.sin(raw_theta - smoothed_theta_rad),
									math.cos(raw_theta - smoothed_theta_rad)
								)

							U, _, Vt = np.linalg.svd(smoothed_R_ref0)
							R_ref0_to_stable = U @ Vt

							scale_factor = smoothed_scale_factor
							theta_rad = smoothed_theta_rad

							calibration_frames_tracked += 1
							map_calibrated = True

							if calibration_frames_tracked == CALIBRATION_LOCK_THRESHOLD:
								node.get_logger().info("🔑 Calibration matrix securely LOCKED and stabilized.")

			if args.show and map_calibrated and args.reference_tag_0 in frame_poses:
				T_cam_ref0 = frame_poses[args.reference_tag_0]
				cos_t = math.cos(theta_rad)
				sin_t = math.sin(theta_rad)
				map_corners = [(0, 0), (args.scale_x, 0), (args.scale_x, args.scale_y), (0, args.scale_y)]
				field_corners_3d = []
				inv_s = 1.0 / scale_factor

				for mx, my in map_corners:
					xf = inv_s * (mx * cos_t + my * sin_t)
					yf = inv_s * (-mx * sin_t + my * cos_t)
					v_stable = np.array([xf, yf, 0.0], dtype=np.float32)
					v_ref0 = R_ref0_to_stable @ v_stable
					p_cam = T_cam_ref0 @ np.append(v_ref0, 1.0)
					field_corners_3d.append(p_cam[0:3])

				camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
				dist_coeffs = np.zeros(4, dtype=np.float32)
				img_pts, _ = cv2.projectPoints(np.array(field_corners_3d, dtype=np.float32), np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs)
				img_pts = img_pts.reshape(-1, 2).astype(np.int32)

				cv2.polylines(frame, [img_pts], True, (255, 0, 255), 2)
				cv2.putText(frame, "Field Plane Grid", tuple(img_pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

			robots_for_ros = []
			robots_for_json = []

			if map_calibrated and args.reference_tag_0 in frame_poses:
				T_cam_ref0 = frame_poses[args.reference_tag_0]
				T_ref0_cam = np.linalg.inv(T_cam_ref0)
				cos_t = math.cos(theta_rad)
				sin_t = math.sin(theta_rad)

				for target_id, robot_id in node.target_tag_map.items():
					if target_id not in frame_poses:
						continue

					T_cam_target = frame_poses[target_id]
					p_target_in_ref0 = T_ref0_cam @ np.append(T_cam_target[0:3, 3], 1.0)
					v_target_stable = R_ref0_to_stable.T @ p_target_in_ref0[0:3]
					x_floor, y_floor = v_target_stable[0], v_target_stable[1]

					raw_x = scale_factor * (x_floor * cos_t - y_floor * sin_t)
					raw_y = scale_factor * (x_floor * sin_t + y_floor * cos_t)

					T_ref0_target = T_ref0_cam @ T_cam_target
					R_stable_target = R_ref0_to_stable.T @ T_ref0_target[0:3, 0:3]

					R_z = np.array([
						[cos_t, -sin_t, 0],
						[sin_t, cos_t, 0],
						[0, 0, 1]
					], dtype=np.float32)
					R_map_target = R_z.T @ R_stable_target
					map_angle_deg = math.degrees(math.atan2(R_map_target[1, 0], R_map_target[0, 0])) % 360.0

					# Apply Filtering
					if target_id in tracked_robots:
						old_x = tracked_robots[target_id]["x"]
						old_y = tracked_robots[target_id]["y"]

						filtered_x = (args.alpha * raw_x) + ((1 - args.alpha) * old_x)
						filtered_y = (args.alpha * raw_y) + ((1 - args.alpha) * old_y)

						current_sin = math.sin(math.radians(map_angle_deg))
						current_cos = math.cos(math.radians(map_angle_deg))

						angle_buffers[target_id]["sin"] = (args.alpha * current_sin) + ((1 - args.alpha) * angle_buffers[target_id]["sin"])
						angle_buffers[target_id]["cos"] = (args.alpha * current_cos) + ((1 - args.alpha) * angle_buffers[target_id]["cos"])

						filtered_rot = math.degrees(math.atan2(angle_buffers[target_id]["sin"], angle_buffers[target_id]["cos"])) % 360
					else:
						filtered_x = raw_x
						filtered_y = raw_y
						filtered_rot = map_angle_deg
						angle_buffers[target_id] = {
							"sin": math.sin(math.radians(map_angle_deg)),
							"cos": math.cos(math.radians(map_angle_deg))
						}

					tracked_robots[target_id] = {
						"x": round(filtered_x, 3),
						"y": round(filtered_y, 3),
						"rotation": round(filtered_rot, 2),
					}

					# Package for output (Note: ROS orientation uses Radians)
					robots_for_ros.append((robot_id, filtered_x, filtered_y, math.radians(filtered_rot)))
					robots_for_json.append({
						"tag_id": target_id,
						"robot_id": robot_id,
						"x": round(filtered_x, 3),
						"y": round(filtered_y, 3),
						"rotation_deg": round(filtered_rot, 2),
						"currently_visible": True
					})

					if args.show and target_id in frame_centers:
						pixel = frame_centers[target_id].astype(int)
						label = f"{robot_id} X:{filtered_x:.1f} Y:{filtered_y:.1f} R:{int(filtered_rot)}d"
						cv2.putText(frame, label, (pixel[0], pixel[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 122, 0), 2)

			for tag_id, pos in tracked_robots.items():
				if tag_id not in frame_poses:
					robot_id = node.target_tag_map.get(tag_id, f"robot_tag_{tag_id}")
					robots_for_json.append({
						"tag_id": tag_id,
						"robot_id": robot_id,
						"x": pos["x"],
						"y": pos["y"],
						"rotation_deg": pos["rotation"],
						"currently_visible": False
					})

			if robots_for_ros:
				node.publish_poses(robots_for_ros)

			output = {
				"event": "frame",
				"visible_tags": sorted(list(frame_centers.keys())),
				"reference_visible": (args.reference_tag_0 in frame_poses and args.reference_tag_1 in frame_poses),
				"map_calibrated": map_calibrated,
				"robots": robots_for_json,
			}
			print(json.dumps(output), flush=True)

			if args.show:
				if not map_calibrated:
					cv2.putText(frame, "Calibration Targets Lost", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
				try:
					cv2.imshow("AprilTag ROS 3D Pose Tracker", frame)
					if cv2.waitKey(1) & 0xFF == ord("q"):
						break
				except cv2.error:
					pass

			output_frame = frame.copy()

			rclpy.spin_once(node, timeout_sec=0.0)
			time.sleep(0.05)

	finally:
		cap.release()
		if args.show:
			try:
				cv2.destroyAllWindows()
			except cv2.error:
				pass
		node.destroy_node()
		rclpy.shutdown()


if __name__ == "__main__":
	main()
