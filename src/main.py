import argparse
import os
import time
from threading import Thread

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

import defines
from locator import Locator

import rclpy
from ros_worker import AprilTagRosTracker, spin_ros_node

import multiprocessing as mp
from stream_server import start_stream_server

def parse_args():
	valid_profiles = [
		attr for attr in dir(defines)
		if not attr.startswith("__") and not hasattr(getattr(defines, attr), "__call__")
	]
	profiles_str = ", ".join(valid_profiles) if valid_profiles else "none found"

	parser = argparse.ArgumentParser(description="Modular AprilTag Tracker")
	parser.add_argument("-c", "--camera-index", type=int, default=int(os.getenv("CAMERA_INDEX", "0")))
	parser.add_argument("-s", "--show", action="store_true", help="Show local debug window")
	parser.add_argument("-t", "--tag-size", type=float, default=float(os.getenv("TAG_SIZE", "0.025")))
	parser.add_argument("-p", "--camera-profile", default=os.getenv("CAMERA_PROFILE", "microsoft_cam"), choices=valid_profiles)
	parser.add_argument("--t0x", type=float, default=float(os.getenv("T0X", "0.0")))
	parser.add_argument("--t0y", type=float, default=float(os.getenv("T0Y", "0.0")))
	parser.add_argument("--t1x", type=float, default=float(os.getenv("T1X", "200.0")))
	parser.add_argument("--t1y", type=float, default=float(os.getenv("T1Y", "200.0")))
	parser.add_argument("-0", "--reference-tag-0", type=int, default=int(os.getenv("REFERENCE_TAG_0", "0")))
	parser.add_argument("-1", "--reference-tag-1", type=int, default=int(os.getenv("REFERENCE_TAG_1", "1")))
	parser.add_argument("--target-tags", default=os.getenv("TARGET_TAGS", "2:robot_1,3:robot_2,4:robot_3,5:robot_4"))
	parser.add_argument("--output-topic", default=os.getenv("OUTPUT_TOPIC", "/cam/pos"))
	parser.add_argument("--publish-individual", action="store_true", help="Publish standalone PoseStamped topics for each robot")
	return parser.parse_args()

def parse_target_tags(raw: str) -> dict:
	result = {}
	for index, item in enumerate(raw.split(","), start=1):
		item = item.strip()
		if not item: continue
		if ":" in item:
			tag_str, robot_id = item.split(":", 1)
			result[int(tag_str.strip())] = robot_id.strip()
		else:
			result[int(item)] = f"robot_{index}"
	return result

def main():
	# Parse arguments and environment variables from start command
	args = parse_args()
	target_tag_map = parse_target_tags(args.target_tags)

	rclpy.init()
	ros_node = AprilTagRosTracker(
		output_topic=args.output_topic,
		publish_individual=args.publish_individual,
		target_tag_map=target_tag_map,
	)

	ros_thread = Thread(target=spin_ros_node, args=(ros_node,), daemon=True)
	ros_thread.start()

	web_queue = None
	if args.show:
		ctx = mp.get_context("spawn")
		web_queue = ctx.Queue(maxsize=2)

		web_process = ctx.Process(
			target=start_stream_server,
			args=(web_queue, "0.0.0.0", 8080),
			daemon=True
		)
		web_process.start()
		print("Web UI broadcast cleanly isolated at http://localhost:8080")

	# Open video stream
	cap = cv2.VideoCapture(args.camera_index)

	cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

	if not cap.isOpened():
		print("Error: Could not open camera {args.camera_index}.")
		ros_node.destroy_node()
		rclpy.shutdown()
		return

	# Create locator for locating AprilTags
	locator = Locator(
		cam_cal=getattr(defines, args.camera_profile),
		tag_size=args.tag_size
	)

	# For saving mapping plane data
	avg_unit_normal_vector = None

	print("Tracking active. Press 'q' in the debug window to exit.")
	try:
		while True:
			# Get a frame
			ret, frame = cap.read()
			if not ret:
				print("Error: Failed to get frame.")
				break

			# Convert to grayscale and get the detected tags with estimated positions
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			detections = locator.detect(gray)
			poses = locator.get_poses(detections)

			# Show the detected tags in an image for debugging purposes
			if args.show:
				for detection in detections:
					tag_id = detection["id"]
					tag_corners = np.array(detection["lb-rb-rt-lt"], dtype=np.int32)
					tag_center = tuple((np.array(detection["center"], dtype=np.float32)).astype(int))

					cv2.polylines(frame, [tag_corners], True, (0, 255, 0), 2)
					cv2.circle(frame, tag_center, 5, (0, 0, 255), -1)
					cv2.putText(frame, str(tag_id), tag_center, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

				frame_downscaled = cv2.resize(frame, (1280, 720), interpolation=cv2.INTER_AREA)

				if web_queue is not None and not web_queue.full():
					try:
						web_queue.put_nowait(frame)
					except Exception:
						pass

			# 1. Extract reference orientations and positions
			normals = []
			pylons = {}

			for tag_id, pose_data in poses.items(): # Get the normals from the reference tags and one of their positions
				if tag_id in (args.reference_tag_0, args.reference_tag_1):
					normals.append(pose_data["R"][:, 2])
					pylons[tag_id] = pose_data["t"]

			if normals:
				# Align and average normals to get a stable ground normal
				reference = normals[0]
				aligned = [n if np.dot(reference, n) >= 0 else -n for n in normals]
				current_avg = np.mean(aligned, axis=0)
				current_avg /= np.linalg.norm(current_avg)

				if avg_unit_normal_vector is not None:
					if np.dot(avg_unit_normal_vector, current_avg) < 0:
						current_avg = -current_avg
				avg_unit_normal_vector = current_avg

			if avg_unit_normal_vector is None:
				print("Looking for reference tags...")
				time.sleep(0.01667)
				continue

			if args.reference_tag_0 in pylons and args.reference_tag_1 in pylons:
				p0_cam = pylons[args.reference_tag_0]
				p1_cam = pylons[args.reference_tag_1]

				delta_p_cam = p1_cam - p0_cam

				dx_field = args.t1x - args.t0x
				dy_field = args.t1y - args.t0y

				delta_p_plane = delta_p_cam - np.dot(delta_p_cam, avg_unit_normal_vector) * avg_unit_normal_vector

				field_link_angle = np.arctan2(dx_field, dy_field)

				c, s = np.cos(-field_link_angle), np.sin(-field_link_angle)

				u = avg_unit_normal_vector
				R_rot = np.array([
					[c + u[0]**2*(1-c),    u[0]*u[1]*(1-c) - u[2]*s, u[0]*u[2]*(1-c) + u[1]*s],
					[u[1]*u[0]*(1-c) + u[2]*s, c + u[1]**2*(1-c),    u[1]*u[2]*(1-c) - u[0]*s],
					[u[2]*u[0]*(1-c) - u[1]*s, u[2]*u[1]*(1-c) + u[0]*s, c + u[2]**2*(1-c)]
				])

				field_north_3d = R_rot @ delta_p_plane
				field_north_3d /= np.linalg.norm(field_north_3d)

				field_east_3d = np.cross(avg_unit_normal_vector, field_north_3d)
				field_east_3d /= np.linalg.norm(field_east_3d)

				pixel_dist = np.linalg.norm(delta_p_plane)
				field_dist = np.sqrt(dx_field**2 + dy_field**2)
				scale_units_per_meter = field_dist / pixel_dist

				field_up_3d = -avg_unit_normal_vector

				R_field_raw = np.column_stack((field_east_3d, field_north_3d, field_up_3d))

				if 'R_field' not in locals():
					R_field = R_field_raw
				else:
					R_field = 0.05 * R_field_raw + 0.95 * R_field
					u_f, _, vh_f = np.linalg.svd(R_field)
					R_field = np.dot(u_f, vh_f)

					if np.linalg.det(R_field) < 0:
						u_f[:, 2] *= -1
						R_field = np.dot(u_f, vh_f)

			else:
				print("System initializing: Please ensure both reference tags are visible.")
				time.sleep(0.01667)
				continue

			if 'R_field' not in locals():
				print("System initializing: Please ensure both reference tags are visible.")
				time.sleep(0.01667)
				continue

			current_frame_robots = []

			for tag_id, pose_data in poses.items():
				if tag_id not in (args.reference_tag_0, args.reference_tag_1):
					if tag_id not in target_tag_map:
						continue  # Skip unmapped tags safely

					# Project the point to the plane
					t_target_cam = pose_data["t"]
					R_target_cam = pose_data["R"]

					v_floor_to_tag = t_target_cam - pylons[args.reference_tag_0]

					live_height_meters = np.dot(v_floor_to_tag, avg_unit_normal_vector)

					corrected_pos_cam = t_target_cam - (live_height_meters * avg_unit_normal_vector)

					v_target_ground_cam = corrected_pos_cam - pylons[args.reference_tag_0]

					pos_x_meters = np.dot(v_target_ground_cam, R_field[:, 0])
					pos_y_meters = np.dot(v_target_ground_cam, R_field[:, 1])

					final_field_x = args.t0x + (pos_x_meters * scale_units_per_meter)
					final_field_y = args.t0y + (pos_y_meters * scale_units_per_meter)

					R_relative = R_field.T @ R_target_cam

					u, _, vh = np.linalg.svd(R_relative)
					R_orthogonal = np.dot(u, vh)

					if np.linalg.det(R_orthogonal) < 0:
						u[:, 2] *= -1
						R_orthogonal = np.dot(u, vh)

					relative_rotation = Rotation.from_matrix(R_orthogonal)
					quaternion_field = relative_rotation.as_quat()

					tag_forward_in_field = R_orthogonal[:, 1]
					theta_rad = np.arctan2(tag_forward_in_field[0], tag_forward_in_field[1]) + np.pi # rotate 180 degrees for up to be 0 degrees
					angle_degrees = np.degrees(theta_rad) % 360.0

					robot_id = target_tag_map[tag_id]
					current_frame_robots.append((robot_id, final_field_x, final_field_y, theta_rad))

					print(f"Chariot tag = {tag_id}, id {robot_id} -> Field Pos: ({final_field_x:.2f}, {final_field_y:.2f}) | Heading: {angle_degrees:.2f}°")



			ros_node.update_robot_data(current_frame_robots)
			ros_node.publish_latest_poses()

			time.sleep(0.01667)

	except KeyboardInterrupt:
		print("\nShutting down pipeline components...")

	finally:
		cap.release()
		cv2.destroyAllWindows()
		ros_node.destroy_node()
		rclpy.shutdown()

if __name__ == "__main__":
	main()


