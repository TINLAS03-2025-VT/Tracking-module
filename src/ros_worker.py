import math
import re
from threading import Lock
from typing import List, Tuple
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose, PoseStamped

def yaw_to_pose_orientation(pose: Pose, yaw_rad: float):
    half = yaw_rad / 2.0
    pose.orientation.x = 0.0
    pose.orientation.y = 0.0
    pose.orientation.z = math.sin(half)
    pose.orientation.w = math.cos(half)

class AprilTagRosTracker(Node):
    def __init__(self, output_topic: str, publish_individual: bool, target_tag_map: dict):
        super().__init__("apriltag_tracker")
        self.output_topic = output_topic
        self.publish_individual = publish_individual
        self.target_tag_map = target_tag_map

        self.pose_array_pub = self.create_publisher(PoseArray, self.output_topic, 10)
        self.individual_pubs = {}   # M.B.: Should get initialized immediately using the given target_tag_map
        self.data_lock = Lock()
        self.latest_robots: List[Tuple[str, float, float, float]] = []

        self.get_logger().info(f"Publishing aggregate poses to {self.output_topic}")
        self.get_logger().info(f"Target tag map: {self.target_tag_map}")

    def update_robot_data(self, robots: List[Tuple[str, float, float, float]]): 
        # M.B.: This helper should be private and used in publish_latest_poses
        # Therefore, whilst this helper is clean, this means is no need for a lock or latest_robots field when the robot poses are immediately used in publish_latest_poses
        # In the main, update_robot_data is also only called once
        with self.data_lock:
            self.latest_robots = robots

    def get_individual_pub(self, robot_id: str): # M.B.: Good helper function, should be used during initialization and set private afterwards
        if robot_id not in self.individual_pubs:
            topic = f"/{robot_id}/pose"
            self.individual_pubs[robot_id] = self.create_publisher(PoseStamped, topic, 10)
            self.get_logger().info(f"Publishing individual pose to {topic}")
        return self.individual_pubs[robot_id]

    def publish_latest_poses(self): # M.B.: This function should use the update_robot_data
        with self.data_lock:
            robots = list(self.latest_robots)

        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = "map"

        for robot_id, x, y, theta in robots:
            pose = Pose()
            pose.position.x = float(x)
            pose.position.y = float(y)

            match = re.search(r'\d+', robot_id)
            if match:
                pose.position.z = float(match.group())
            else:
                pose.position.z = 0.0

            yaw_to_pose_orientation(pose, theta)
            pose_array.poses.append(pose)

            if self.publish_individual:
                stamped = PoseStamped()
                stamped.header = pose_array.header
                stamped.pose = pose
                self.get_individual_pub(robot_id).publish(stamped)

        if robots:
            self.pose_array_pub.publish(pose_array)

def spin_ros_node(node): # M.B.: Should be in the AprilTagRosTracker class itself
    rclpy.spin(node)
