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
from threaded_cam import ThreadedCamera

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

def project_field_points_to_pixel(points_field, p0_cam, R_field, scale_units_per_meter, args, cam_cal):
	"""
	Projects an array of field points [[xf1, yf1], [xf2, yf2], ...] to pixel coordinates.
	"""
	points_field = np.array(points_field, dtype=np.float32)

	# Translate from field origin and scale to meters
	pos_x_meters = (points_field[:, 0] - args.t0x) / scale_units_per_meter
	pos_y_meters = (points_field[:, 1] - args.t0y) / scale_units_per_meter

	# Vectorized 3D conversion in camera space
	v_cam = (pos_x_meters[:, None] * R_field[:, 0]) + \
			(pos_y_meters[:, None] * R_field[:, 1]) + p0_cam

	# Camera intrinsics matrix
	mtx = np.array([
		[cam_cal["fx"], 0, cam_cal["cx"]],
		[0, cam_cal["fy"], cam_cal["cy"]],
		[0, 0, 1]
	], dtype=np.float32)
	dist = np.zeros(5, dtype=np.float32)

	# Project all points at once using OpenCV's optimized backend
	pts_2d, _ = cv2.projectPoints(
		v_cam.astype(np.float32),
		np.zeros(3), np.zeros(3), mtx, dist
	)

	# Return as an integer array of shape (N, 2) pixels
	return pts_2d.reshape(-1, 2).astype(int)

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
	cap = ThreadedCamera(args.camera_index, width=1280, height=720)
	cap.start()
	time.sleep(0.5)

	ret, initial_frame = cap.read()
	if not ret:
		print(f"Error: Could not pull initial frame from camera {args.camera_index}.")
		cap.release()
		ros_node.destroy_node()
		rclpy.shutdown()
		return

	cam_cal_profile = getattr(defines, args.camera_profile)

	# Create locator for locating AprilTags
	locator = Locator(
		cam_cal=cam_cal_profile,
		tag_size=args.tag_size
	)

	# --- Calibration & Smoothing Storage Variables ---
	calibrated = False
	calibration_frames_gathered = 0
	REQUIRED_CAL_FRAMES = 100

	# Accumulators for averaging
	accumulated_normals = []
	accumulated_p0 = []
	accumulated_p1 = []

	# Final static locked coordinate transform properties
	locked_avg_normal = None
	locked_p0_cam = None
	locked_R_field = None
	locked_scale = None

	# Position smoothing dictionary for active targets
	field_smoothers = {}
	POS_ALPHA = 0.25

	last_perf_print = time.time()

	print("Tracking active. Press 'q' in the debug window to exit.")
	try:
		while True:
			# loop_start = time.time()
			t0 = time.perf_counter()
			# Get a frame
			ret, frame_raw = cap.read()
			if not ret:
				print("Error: Failed to get frame.")
				# elapsed = time.time() - loop_start
				# sleep_time = max(0.0, 0.01667 - elapsed)
				# time.sleep(sleep_time)
				break
			# print("Got frame!")
			t1 = time.perf_counter()

			frame = frame_raw # cv2.resize(frame_raw, (1280, 720), interpolation=cv2.INTER_LINEAR)
			t2 = time.perf_counter()

			# Convert to grayscale and get the detected tags with estimated positions
			gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
			# print("Created raw frame!")
			t3 = time.perf_counter()

			if calibrated:
				wanted_tag_ids = set(target_tag_map.keys())
			else:
				wanted_tag_ids = {args.reference_tag_0, args.reference_tag_1}

			detections = locator.detect(gray, wanted_tag_ids=wanted_tag_ids)
			# print("Got detections!")
			t4 = time.perf_counter()

			poses = locator.get_poses(detections, wanted_tag_ids=wanted_tag_ids)
			# print("Got poses!")
			t5 = time.perf_counter()

			# Show the detected tags in an image for debugging purposes
			if args.show:
				# Draw detections
				for detection in detections:
					tag_id = detection["id"]
					tag_corners = np.array(detection["lb-rb-rt-lt"], dtype=np.int32)
					tag_center = tuple((np.array(detection["center"], dtype=np.float32)).astype(int))

					cv2.polylines(frame, [tag_corners], True, (0, 255, 0), 2)
					cv2.circle(frame, tag_center, 5, (0, 0, 255), -1)
					cv2.putText(frame, str(tag_id), tag_center, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

				if calibrated:
					min_x, max_x = min(args.t0x, args.t1x), max(args.t0x, args.t1x)
					min_y, max_y = min(args.t0y, args.t1y), max(args.t0y, args.t1y)
					grid_color = (100, 100, 100)

					# 1. Generate all grid line boundaries
					x_vals = list(range(int(np.ceil(min_x)), int(np.floor(max_x)) + 1))
					y_vals = list(range(int(np.ceil(min_y)), int(np.floor(max_y)) + 1))

					grid_points = []

					# Append start/end pairs for X lines
					for x_val in x_vals:
						grid_points.append([x_val, min_y])
						grid_points.append([x_val, max_y])

					# Append start/end pairs for Y lines
					for y_val in y_vals:
						grid_points.append([min_x, y_val])
						grid_points.append([max_x, y_val])

					# Append outer boundary corner points (4 corners)
					f_corners = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]
					grid_points.extend(f_corners)

					# 2. Project EVERY point in a single batch call
					all_pixels = project_field_points_to_pixel(
						grid_points, locked_p0_cam, locked_R_field, locked_scale, args, cam_cal_profile
					)

					# 3. Dissect the single flat pixel array back into drawing segments
					idx = 0

					# Draw X lines
					for x_val in x_vals:
						p_start, p_end = tuple(all_pixels[idx]), tuple(all_pixels[idx+1])
						cv2.line(frame, p_start, p_end, grid_color, 1, cv2.LINE_AA)
						cv2.putText(frame, f"X={x_val}", (p_start[0], p_start[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, grid_color, 1, cv2.LINE_AA)
						idx += 2

					# Draw Y lines
					for y_val in y_vals:
						p_start, p_end = tuple(all_pixels[idx]), tuple(all_pixels[idx+1])
						cv2.line(frame, p_start, p_end, grid_color, 1, cv2.LINE_AA)
						cv2.putText(frame, f"Y={y_val}", (p_start[0] + 5, p_start[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.35, grid_color, 1, cv2.LINE_AA)
						idx += 2

					# Draw outer thick boundaries
					pixel_corners = all_pixels[idx:idx+4]
					boundary_color = (0, 255, 255)
					for i in range(4):
						pA = tuple(pixel_corners[i])
						pB = tuple(pixel_corners[(i + 1) % 4])
						cv2.line(frame, pA, pB, boundary_color, 2, cv2.LINE_AA)

				if web_queue is not None:
					try:
						if web_queue.full():
							try:
								web_queue.get_nowait()
							except Exception:
								pass

						web_queue.put_nowait(frame)
					except Exception:
						pass
			tshow = time.perf_counter()

			# 1. Extract reference orientations and positions
			current_pylons = {}
			current_normals = []

			for tag_id, pose_data in poses.items(): # Get the normals from the reference tags and one of their positions
				if tag_id in (args.reference_tag_0, args.reference_tag_1):
					current_normals.append(pose_data["R"][:, 2])
					current_pylons[tag_id] = pose_data["t"]

			if not calibrated:
				if args.reference_tag_0 in current_pylons and args.reference_tag_1 in current_pylons:
					accumulated_p0.append(current_pylons[args.reference_tag_0])
					accumulated_p1.append(current_pylons[args.reference_tag_1])

					# Gather tracking reference orientations
					ref = current_normals[0]
					aligned = [n if np.dot(ref, n) >= 0 else -n for n in current_normals]
					accumulated_normals.append(np.mean(aligned, axis=0))

					calibration_frames_gathered += 1
					if calibration_frames_gathered % 20 == 0:
						print(f"Calibrating field structure: {calibration_frames_gathered}/{REQUIRED_CAL_FRAMES} frames...")
				else:
					print("Calibration holding: Both reference tags must be cleanly visible.")

					# elapsed = time.time() - loop_start
					# sleep_time = max(0.0, 0.01667 - elapsed)
					# time.sleep(sleep_time)
					continue

				if calibration_frames_gathered >= REQUIRED_CAL_FRAMES:
					# Compute robust locked coordinate space rules
					locked_p0_cam = np.mean(accumulated_p0, axis=0)
					mean_p1_cam = np.mean(accumulated_p1, axis=0)

					locked_avg_normal = np.mean(accumulated_normals, axis=0)
					locked_avg_normal /= np.linalg.norm(locked_avg_normal)

					delta_p_cam = mean_p1_cam - locked_p0_cam
					dx_field = args.t1x - args.t0x
					dy_field = args.t1y - args.t0y

					delta_p_plane = delta_p_cam - np.dot(delta_p_cam, locked_avg_normal) * locked_avg_normal
					field_link_angle = np.arctan2(dx_field, dy_field)

					c, s = np.cos(-field_link_angle), np.sin(-field_link_angle)
					u = locked_avg_normal
					R_rot = np.array([
						[c + u[0]**2*(1-c),    u[0]*u[1]*(1-c) - u[2]*s, u[0]*u[2]*(1-c) + u[1]*s],
						[u[1]*u[0]*(1-c) + u[2]*s, c + u[1]**2*(1-c),    u[1]*u[2]*(1-c) - u[0]*s],
						[u[2]*u[0]*(1-c) - u[1]*s, u[2]*u[1]*(1-c) + u[0]*s, c + u[2]**2*(1-c)]
					])

					field_north_3d = R_rot @ delta_p_plane
					field_north_3d /= np.linalg.norm(field_north_3d)

					field_east_3d = np.cross(locked_avg_normal, field_north_3d)
					field_east_3d /= np.linalg.norm(field_east_3d)

					pixel_dist = np.linalg.norm(delta_p_plane)
					field_dist = np.sqrt(dx_field**2 + dy_field**2)
					locked_scale = field_dist / pixel_dist

					field_up_3d = -locked_avg_normal
					R_field_raw = np.column_stack((field_east_3d, field_north_3d, field_up_3d))

					u_f, _, vh_f = np.linalg.svd(R_field_raw)
					locked_R_field = np.dot(u_f, vh_f)
					if np.linalg.det(locked_R_field) < 0:
						u_f[:, 2] *= -1
						locked_R_field = np.dot(u_f, vh_f)

					calibrated = True
					print(">>> FIELD CALIBRATION LOCKED SUCCESSFUL. Tracking active. <<<")

				# elapsed = time.time() - loop_start
				# sleep_time = max(0.0, 0.01667 - elapsed)
				# time.sleep(sleep_time)
				continue

			current_frame_robots = []

			for tag_id, pose_data in poses.items():
				if tag_id not in (args.reference_tag_0, args.reference_tag_1):
					if tag_id not in target_tag_map:
						continue  # Skip unmapped tags safely

					# Project the point to the plane
					t_target_cam = pose_data["t"]
					R_target_cam = pose_data["R"]

					v_floor_to_tag = t_target_cam - locked_p0_cam

					live_height_meters = np.dot(v_floor_to_tag, locked_avg_normal)

					corrected_pos_cam = t_target_cam - (live_height_meters * locked_avg_normal)

					v_target_ground_cam = corrected_pos_cam - locked_p0_cam

					pos_x_meters = np.dot(v_target_ground_cam, locked_R_field[:, 0])
					pos_y_meters = np.dot(v_target_ground_cam, locked_R_field[:, 1])

					raw_field_x = args.t0x + (pos_x_meters * locked_scale)
					raw_field_y = args.t0y + (pos_y_meters * locked_scale)

					if tag_id not in field_smoothers:
						field_smoothers[tag_id] = (raw_field_x, raw_field_y)

					prev_smooth_x, prev_smooth_y = field_smoothers[tag_id]
					final_field_x = (POS_ALPHA * raw_field_x) + ((1.0 - POS_ALPHA) * prev_smooth_x)
					final_field_y = (POS_ALPHA * raw_field_y) + ((1.0 - POS_ALPHA) * prev_smooth_y)
					field_smoothers[tag_id] = (final_field_x, final_field_y)

					R_relative = locked_R_field.T @ R_target_cam

					# u, _, vh = np.linalg.svd(R_relative)
					# R_orthogonal = np.dot(u, vh)
     #
					# if np.linalg.det(R_orthogonal) < 0:
					# 	u[:, 2] *= -1
					# 	R_orthogonal = np.dot(u, vh)
					R_orthogonal = R_relative

					tag_forward_in_field = R_orthogonal[:, 1]
					theta_rad = np.arctan2(-tag_forward_in_field[1], -tag_forward_in_field[0])
					angle_degrees = np.degrees(theta_rad)

					robot_id = target_tag_map[tag_id]
					current_frame_robots.append((robot_id, final_field_x, final_field_y, theta_rad))

					# print(f"Chariot tag = {tag_id}, id {robot_id} -> Field Pos: ({final_field_x:.2f}, {final_field_y:.2f}) | Heading: {angle_degrees:.2f}°")

			t6 = time.perf_counter()

			ros_node.update_robot_data(current_frame_robots)
			ros_node.publish_latest_poses()
			t7 = time.perf_counter()

			if time.time() - last_perf_print > 1.0:
				print(
					f"[PERF] read={(t1-t0)*1000:.1f}ms "
					f"downscale ={(t2-t1)*1000:.1f}ms "
					f"gray={(t3-t2)*1000:.1f}ms "
					f"detect={(t4-t3)*1000:.1f}ms "
					f"poses={(t5-t4)*1000:.1f}ms "
					f"show={(tshow-t5)*1000:.1f}ms "
					f"calc={(t6-tshow)*1000:.1f}ms "
					f"ros={(t7-t6)*1000:.1f}ms "
					f"total={(t7-t0)*1000:.1f}ms"
				)
				last_perf_print = time.time()


			# elapsed = time.time() - loop_start
			# sleep_time = max(0.0, 0.01667 - elapsed)
			# time.sleep(sleep_time)

	except KeyboardInterrupt:
		print("\nShutting down pipeline components...")

	finally:
		cap.release()
		ros_node.destroy_node()
		rclpy.shutdown()

if __name__ == "__main__":
	main()


