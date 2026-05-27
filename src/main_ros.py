import argparse
import json
import math
import os
import time
from typing import Dict, List, Tuple

import apriltag
import cv2
import numpy as np

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose, PoseStamped

import defines


def parse_args():
    parser = argparse.ArgumentParser(description="AprilTag tracker with ROS 2 publishing")
    parser.add_argument("--camera-index", type=int, default=int(os.getenv("CAMERA_INDEX", "0")))
    parser.add_argument("--show", action="store_true", help="Show OpenCV debug window")
    parser.add_argument("--scale-x", type=float, default=float(os.getenv("SCALE_X", "200")))
    parser.add_argument("--scale-y", type=float, default=float(os.getenv("SCALE_Y", "200")))
    parser.add_argument("--reference-tag-0", type=int, default=int(os.getenv("REFERENCE_TAG_0", "0")))
    parser.add_argument("--reference-tag-1", type=int, default=int(os.getenv("REFERENCE_TAG_1", "1")))
    parser.add_argument("--target-tags", default=os.getenv("TARGET_TAGS", "2:robot_1,3:robot_2"))
    parser.add_argument("--camera-profile", default=os.getenv("CAMERA_PROFILE", "microsoft_cam"))
    parser.add_argument("--output-topic", default=os.getenv("OUTPUT_TOPIC", "/robots/pos"))
    parser.add_argument(
        "--publish-individual-poses",
        default=os.getenv("PUBLISH_INDIVIDUAL_POSES", "true").lower() == "true",
        action=argparse.BooleanOptionalAction,
    )
    return parser.parse_args()


def parse_target_tags(raw: str) -> Dict[int, str]:
    """
    Supports:
      TARGET_TAGS=2,3
      TARGET_TAGS=2:robot_1,3:robot_2
    """
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


def get_center(det):
    return np.array(det["center"], dtype=np.float32)


def get_corners(det):
    return np.array(det["lb-rb-rt-lt"], dtype=np.float32)


def build_transform(frame_tags, reference_tag_0, reference_tag_1, scale_x, scale_y):
    if reference_tag_0 not in frame_tags or reference_tag_1 not in frame_tags:
        return None

    p0 = np.array(frame_tags[reference_tag_0], dtype=np.float32)
    p1 = np.array(frame_tags[reference_tag_1], dtype=np.float32)

    v = p1 - p0
    p2 = p0 + np.array([-v[1], v[0]], dtype=np.float32)

    src_pts = np.array([p0, p1, p2], dtype=np.float32)

    dest_0 = np.array([0.0, 0.0], dtype=np.float32)
    dest_1 = np.array([scale_x, scale_y], dtype=np.float32)

    v_dest = dest_1 - dest_0
    dest_2 = dest_0 + np.array([-v_dest[1], v_dest[0]], dtype=np.float32)

    dest_pts = np.array([dest_0, dest_1, dest_2], dtype=np.float32)

    return cv2.getAffineTransform(src_pts, dest_pts)


def apply_transform(matrix, pixel):
    homogeneous = np.array([pixel[0], pixel[1], 1.0], dtype=np.float32)
    mapped = np.dot(matrix, homogeneous)
    return float(mapped[0]), float(mapped[1])


def estimate_theta(matrix, det) -> float:
    corners = get_corners(det)

    # corners are lb, rb, rt, lt
    lb = corners[0]
    rb = corners[1]

    x1, y1 = apply_transform(matrix, lb)
    x2, y2 = apply_transform(matrix, rb)

    return math.atan2(y2 - y1, x2 - x1)


def yaw_to_pose_orientation(pose: Pose, yaw: float):
    half = yaw / 2.0
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
    args = parse_args()

    # Keep this validation, so config mistakes fail early.
    _ = get_camera_calibration(args.camera_profile)

    rclpy.init()
    node = AprilTagRosTracker(args)

    detector = apriltag.apriltag(
        family="tagStandard41h12",
        threads=4,
        maxhamming=1,
        decimate=2.0,
        blur=0.0,
        refine_edges=True,
        debug=False,
    )

    cap = cv2.VideoCapture(args.camera_index)

    if not cap.isOpened():
        node.get_logger().error(f"Could not open camera index {args.camera_index}")
        raise RuntimeError(f"Could not open camera index {args.camera_index}")

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
                node.get_logger().warn("Could not read frame from camera")
                time.sleep(0.2)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = detector.detect(gray)

            frame_tags = {}
            detections_by_id = {}

            for det in detections:
                tag_id = int(det["id"])
                center = get_center(det)
                frame_tags[tag_id] = center
                detections_by_id[tag_id] = det

                if args.show:
                    corners = get_corners(det).astype(int)
                    center_int = tuple(center.astype(int))
                    cv2.polylines(frame, [corners], True, (0, 255, 0), 2)
                    cv2.circle(frame, center_int, 5, (0, 0, 255), -1)
                    cv2.putText(frame, str(tag_id), center_int, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            transform = build_transform(
                frame_tags,
                args.reference_tag_0,
                args.reference_tag_1,
                args.scale_x,
                args.scale_y,
            )

            robots_for_ros = []
            robots_for_json = []

            if transform is not None:
                for target_id, robot_id in node.target_tag_map.items():
                    if target_id not in frame_tags:
                        continue

                    x, y = apply_transform(transform, frame_tags[target_id])
                    theta = estimate_theta(transform, detections_by_id[target_id])

                    robots_for_ros.append((robot_id, x, y, theta))
                    robots_for_json.append({
                        "tag_id": target_id,
                        "robot_id": robot_id,
                        "x": round(x, 3),
                        "y": round(y, 3),
                        "theta": round(theta, 3),
                    })

                    if args.show:
                        pixel = frame_tags[target_id].astype(int)
                        cv2.putText(
                            frame,
                            f"{robot_id}: ({int(x)}, {int(y)})",
                            (int(pixel[0]), int(pixel[1]) + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 255),
                            1,
                        )

            node.publish_poses(robots_for_ros)

            print(json.dumps({
                "event": "frame",
                "visible_tags": sorted(list(frame_tags.keys())),
                "reference_visible": transform is not None,
                "robots": robots_for_json,
            }), flush=True)

            if args.show:
                if transform is None:
                    cv2.putText(
                        frame,
                        "Reference tags missing",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

                cv2.imshow("AprilTag ROS Tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(0.05)

    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
