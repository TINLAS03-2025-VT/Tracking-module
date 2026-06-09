import argparse
import json
import math
import os
import time
from threading import Thread
import cv2

import rclpy
import defines
from ros_worker import AprilTagRosTracker, spin_ros_node
from web_server import run_flask, update_web_frame
from tracker_engine import AprilTagTrackerEngine

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
    parser.add_argument("-x", "--scale-x", type=float, default=float(os.getenv("SCALE_X", "200")))
    parser.add_argument("-y", "--scale-y", type=float, default=float(os.getenv("SCALE_Y", "200")))
    parser.add_argument("-0", "--reference-tag-0", type=int, default=int(os.getenv("REFERENCE_TAG_0", "0")))
    parser.add_argument("-1", "--reference-tag-1", type=int, default=int(os.getenv("REFERENCE_TAG_1", "1")))
    parser.add_argument("-a", "--alpha", type=float, default=float(os.getenv("TRACKER_ALPHA", "0.3")))
    parser.add_argument("--target-tags", default=os.getenv("TARGET_TAGS", "2:robot_1,3:robot_2"))
    parser.add_argument("--output-topic", default=os.getenv("OUTPUT_TOPIC", "/robots/pos"))
    parser.add_argument("--publish-individual-poses", default=os.getenv("PUBLISH_INDIVIDUAL_POSES", "true").lower() == "true", action=argparse.BooleanOptionalAction)
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
    args = parse_args()
    cam_cal = getattr(defines, args.camera_profile)

    # Initialize Core Tracking Components
    engine = AprilTagTrackerEngine(
        args.tag_size, args.alpha, args.scale_x, args.scale_y,
        args.reference_tag_0, args.reference_tag_1,
        cam_cal["fx"], cam_cal["fy"], cam_cal["cx"], cam_cal["cy"]
    )

    # Initialize ROS2 Engine Thread
    rclpy.init()
    target_tag_map = parse_target_tags(args.target_tags)
    ros_node = AprilTagRosTracker(args.output_topic, args.publish_individual_poses, target_tag_map)
    Thread(target=spin_ros_node, args=(ros_node,), daemon=True).start()

    # Initialize Web App Thread
    Thread(target=run_flask, daemon=True).start()

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        ros_node.get_logger().error(f"Could not open camera index {args.camera_index}")
        return

    print(json.dumps({"event": "tracker_running", "target_tag_map": target_tag_map}), flush=True)

    try:
        while rclpy.ok():
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.2)
                continue

            # Convert frame to Grayscale, then construct colored canvas over top
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            annotated_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

            # Process tracking steps via extracted engine module
            detections, frame_poses, frame_centers = engine.process_frame(gray)
            engine.compute_calibration(frame_poses, logger=ros_node.get_logger())

            # Draw Detections in Color
            for det in detections:
                tag_id = int(det["id"])
                if tag_id in frame_centers:
                    corners = np.array(det["lb-rb-rt-lt"], dtype=np.int32)
                    center_int = tuple(frame_centers[tag_id].astype(int))
                    cv2.polylines(annotated_frame, [corners], True, (0, 255, 0), 2)
                    cv2.circle(annotated_frame, center_int, 5, (0, 0, 255), -1)
                    cv2.putText(annotated_frame, str(tag_id), center_int, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Draw Ground Map Plane Grid Line
            grid_pts = engine.get_field_grid_points(frame_poses)
            if grid_pts is not None:
                cv2.polylines(annotated_frame, [grid_pts], True, (255, 0, 255), 2)
                cv2.putText(annotated_frame, "Field Plane Grid", tuple(grid_pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

            robots_for_ros = []
            robots_for_json = []

            # Compute actual positions
            for target_id, robot_id in ros_node.target_tag_map.items():
                loc = engine.localize_target(target_id, frame_poses)
                if loc is not None:
                    fx_m, fy_m, frot_deg = loc
                    robots_for_ros.append((robot_id, fx_m, fy_m, math.radians(frot_deg)))
                    robots_for_json.append({
                        "tag_id": target_id, "robot_id": robot_id, "x": round(fx_m, 3), "y": round(fy_m, 3),
                        "rotation_deg": round(frot_deg, 2), "currently_visible": True
                    })

                    if target_id in frame_centers:
                        pixel = frame_centers[target_id].astype(int)
                        label = f"{robot_id} X:{fx_m:.1f} Y:{fy_m:.1f} R:{int(frot_deg)}d"
                        cv2.putText(annotated_frame, label, (pixel[0], pixel[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 122, 0), 2)

            # Add missing historical context metrics to json outputs
            for tag_id, pos in engine.tracked_robots.items():
                if tag_id not in frame_poses:
                    r_id = ros_node.target_tag_map.get(tag_id, f"robot_tag_{tag_id}")
                    robots_for_json.append({
                        "tag_id": tag_id, "robot_id": r_id, "x": pos["x"], "y": pos["y"],
                        "rotation_deg": pos["rotation"], "currently_visible": False
                    })

            # Send state sync signals down pipeline pipelines
            ros_node.update_robot_data(robots_for_ros)
            ros_node.publish_latest_poses()

            print(json.dumps({
                "event": "frame", "visible_tags": sorted(list(frame_centers.keys())),
                "reference_visible": (args.reference_tag_0 in frame_poses and args.reference_tag_1 in frame_poses),
                "map_calibrated": engine.map_calibrated, "robots": robots_for_json
            }), flush=True)

            if not engine.map_calibrated:
                cv2.putText(annotated_frame, "Calibration Targets Lost", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if args.show:
                cv2.imshow("AprilTag ROS 3D Pose Tracker", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            update_web_frame(annotated_frame)
            time.sleep(0.05)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        ros_node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()