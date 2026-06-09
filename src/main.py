import argparse
import json
import math
import os
import time
from threading import Thread
import cv2
import numpy as np
import apriltag

import rclpy
import defines
from ros_worker import AprilTagRosTracker, spin_ros_node
from web_server import run_flask, update_web_frame

def parse_args():
    valid_profiles = [
        attr for attr in dir(defines)
        if not attr.startswith("__") and not hasattr(getattr(defines, attr), "__call__")
    ]
    profiles_str = ", ".join(valid_profiles) if valid_profiles else "none found"

    parser = argparse.ArgumentParser(description="Headless AprilTag tracker")
    parser.add_argument("-c", "--camera-index", type=int, default=int(os.getenv("CAMERA_INDEX", "0")), help="index of the camera to use")
    parser.add_argument("-s", "--show", action="store_true", help="show OpenCV debug window")
    parser.add_argument("-t", "--tag-size", type=float, default=float(os.getenv("TAG_SIZE", "0.025")), help="inner dimensions of the tags in meters")
    parser.add_argument("-p", "--camera-profile", default=os.getenv("CAMERA_PROFILE", "microsoft_cam"), choices=valid_profiles, help=f"camera profile options: {profiles_str}")
    parser.add_argument("-x", "--scale-x", type=float, default=float(os.getenv("SCALE_X", "200")), help="x-axis scale to map the locations to")
    parser.add_argument("-y", "--scale-y", type=float, default=float(os.getenv("SCALE_Y", "200")), help="y-axis scale to map the locations to")
    parser.add_argument("-0", "--reference-tag-0", type=int, default=int(os.getenv("REFERENCE_TAG_0", "0")), help="tag ID for reference tag at location {x: 0, y: 0}")
    parser.add_argument("-1", "--reference-tag-1", type=int, default=int(os.getenv("REFERENCE_TAG_1", "1")), help="tag ID for reference tag at location {x: SCALE_X, y: SCALE_Y}")
    parser.add_argument("-a", "--alpha", type=float, default=float(os.getenv("TRACKER_ALPHA", "0.3")), help="weight alpha for exponential moving average")
    parser.add_argument("--target-tags", default=os.getenv("TARGET_TAGS", "2:robot_1,3:robot_2"), help="Mapping target tags to robot names")
    parser.add_argument("--output-topic", default=os.getenv("OUTPUT_TOPIC", "/robots/pos"), help="ROS 2 topic for aggregate poses")
    parser.add_argument("--publish-individual-poses", default=os.getenv("PUBLISH_INDIVIDUAL_POSES", "true").lower() == "true", action=argparse.BooleanOptionalAction, help="Publish independent PoseStamped topics for each robot")
    return parser.parse_args()

def parse_target_tags(raw: str) -> dict:
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

def make_4x4_matrix(R, t):
    T = np.eye(4, dtype=np.float32)
    T[0:3, 0:3] = R
    T[0:3, 3] = t.flatten()
    return T

def main():
    args = parse_args()
    cam_cal = getattr(defines, args.camera_profile)
    fx, fy, cx, cy = cam_cal["fx"], cam_cal["fy"], cam_cal["cx"], cam_cal["cy"]

    # --- Initialize ROS2 Threading Environment ---
    rclpy.init()
    target_tag_map = parse_target_tags(args.target_tags)
    ros_node = AprilTagRosTracker(args.output_topic, args.publish_individual_poses, target_tag_map)

    ros_thread = Thread(target=spin_ros_node, args=(ros_node,), daemon=True)
    ros_thread.start()

    # --- Initialize Web Server App Threading Environment ---
    flask_thread = Thread(target=run_flask, daemon=True)
    flask_thread.start()

    # Setup Vision Processing Objects
    detector = apriltag.apriltag(family="tagStandard41h12", threads=8, refine_edges=True, debug=False)
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        ros_node.get_logger().error(f"Could not open camera index {args.camera_index}")
        raise RuntimeError(f"Could not open camera index {args.camera_index}")

    tracked_robots = {}
    angle_buffers = {}
    map_calibrated = False
    R_ref0_to_stable = None
    scale_factor, theta_rad = 1.0, 0.0
    smoothed_R_ref0, smoothed_scale_factor, smoothed_theta_rad = None, 1.0, 0.0
    calibration_frames_tracked, CALIBRATION_LOCK_THRESHOLD = 0, 50

    print(json.dumps({
        "event": "ros_tracker_started",
        "camera_index": args.camera_index,
        "target_tag_map": target_tag_map,
    }), flush=True)

    try:
        while rclpy.ok():
            ret, frame = cap.read()
            if not ret:
                ros_node.get_logger().warn("Camera frame dropped.")
                time.sleep(0.2)
                continue

            # Work on frame detections
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = detector.detect(gray)

            frame_poses = {}
            frame_centers = {}
            annotated_frame = frame.copy() # Store annotations cleanly

            for det in detections:
                tag_id = int(det["id"])
                center = np.array(det["center"], dtype=np.float32)
                frame_centers[tag_id] = center

                pose = detector.estimate_tag_pose(det, args.tag_size, fx, fy, cx, cy)
                frame_poses[tag_id] = make_4x4_matrix(pose['R'], pose['t'])

                # Render Colored Annotations directly over our target copy
                corners = np.array(det["lb-rb-rt-lt"], dtype=np.int32)
                center_int = tuple(center.astype(int))
                cv2.polylines(annotated_frame, [corners], True, (0, 255, 0), 2)
                cv2.circle(annotated_frame, center_int, 5, (0, 0, 255), -1)
                cv2.putText(annotated_frame, str(tag_id), center_int, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # --- Calibrate/Coordinate Space Mapping ---
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
                        u_y_ref0 = np.cross(n_ref0, v_ref0)
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
                                ros_node.get_logger().info("🔑 Calibration matrix securely LOCKED.")

            # Draw Field Plane Grid
            if map_calibrated and args.reference_tag_0 in frame_poses:
                T_cam_ref0 = frame_poses[args.reference_tag_0]
                cos_t, sin_t = math.cos(theta_rad), math.sin(theta_rad)
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

                cv2.polylines(annotated_frame, [img_pts], True, (255, 0, 255), 2)
                cv2.putText(annotated_frame, "Field Plane Grid", tuple(img_pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            robots_for_ros = []
            robots_for_json = []

            if map_calibrated and args.reference_tag_0 in frame_poses:
                T_cam_ref0 = frame_poses[args.reference_tag_0]
                T_ref0_cam = np.linalg.inv(T_cam_ref0)
                cos_t, sin_t = math.cos(theta_rad), math.sin(theta_rad)

                for target_id, robot_id in ros_node.target_tag_map.items():
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

                    R_z = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]], dtype=np.float32)
                    R_map_target = R_z.T @ R_stable_target
                    map_angle_deg = math.degrees(math.atan2(R_map_target[1, 0], R_map_target[0, 0])) % 360.0

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
                        filtered_x, filtered_y, filtered_rot = raw_x, raw_y, map_angle_deg
                        angle_buffers[target_id] = {
                            "sin": math.sin(math.radians(map_angle_deg)),
                            "cos": math.cos(math.radians(map_angle_deg))
                        }

                    tracked_robots[target_id] = {
                        "x": round(filtered_x, 3), "y": round(filtered_y, 3), "rotation": round(filtered_rot, 2)
                    }

                    robots_for_ros.append((robot_id, filtered_x, filtered_y, math.radians(filtered_rot)))
                    robots_for_json.append({
                        "tag_id": target_id, "robot_id": robot_id,
                        "x": round(filtered_x, 3), "y": round(filtered_y, 3),
                        "rotation_deg": round(filtered_rot, 2), "currently_visible": True
                    })

                    if target_id in frame_centers:
                        pixel = frame_centers[target_id].astype(int)
                        label = f"{robot_id} X:{filtered_x:.1f} Y:{filtered_y:.1f} R:{int(filtered_rot)}d"
                        cv2.putText(annotated_frame, label, (pixel[0], pixel[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 122, 0), 2)

            for tag_id, pos in tracked_robots.items():
                if tag_id not in frame_poses:
                    robot_id = ros_node.target_tag_map.get(tag_id, f"robot_tag_{tag_id}")
                    robots_for_json.append({
                        "tag_id": tag_id, "robot_id": robot_id, "x": pos["x"], "y": pos["y"],
                        "rotation_deg": pos["rotation"], "currently_visible": False
                    })

            # Update the shared thread state structures
            ros_node.update_robot_data(robots_for_ros)
            ros_node.publish_latest_poses()

            print(json.dumps({
                "event": "frame", "visible_tags": sorted(list(frame_centers.keys())),
                "reference_visible": (args.reference_tag_0 in frame_poses and args.reference_tag_1 in frame_poses),
                "map_calibrated": map_calibrated, "robots": robots_for_json,
            }), flush=True)

            if not map_calibrated:
                cv2.putText(annotated_frame, "Calibration Targets Lost", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if args.show:
                try:
                    cv2.imshow("AprilTag ROS 3D Pose Tracker", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                except cv2.error:
                    pass

            # Grayscale video conversion inside this pipeline utility
            update_web_frame(annotated_frame)
            time.sleep(0.05)

    finally:
        cap.release()
        if args.show:
            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass
        ros_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()