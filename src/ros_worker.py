import math
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
        self.individual_pubs = {}
        self.data_lock = Lock()
        self.latest_robots: List[Tuple[str, float, float, float]] = []

        self.get_logger().info(f"Publishing aggregate poses to {self.output_topic}")
        self.get_logger().info(f"Target tag map: {self.target_tag_map}")

    def update_robot_data(self, robots: List[Tuple[str, float, float, float]]):
        with self.data_lock:
            self.latest_robots = robots

    def get_individual_pub(self, robot_id: str):
        if robot_id not in self.individual_pubs:
            topic = f"/{robot_id}/pose"
            self.individual_pubs[robot_id] = self.create_publisher(PoseStamped, topic, 10)
            self.get_logger().info(f"Publishing individual pose to {topic}")
        return self.individual_pubs[robot_id]

    def publish_latest_poses(self):
        with self.data_lock:
            robots = list(self.latest_robots)

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

            if self.publish_individual:
                stamped = PoseStamped()
                stamped.header = pose_array.header
                stamped.pose = pose
                self.get_individual_pub(robot_id).publish(stamped)

        if robots:
            self.pose_array_pub.publish(pose_array)

def spin_ros_node(node):
    rclpy.spin(node)