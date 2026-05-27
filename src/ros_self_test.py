import time

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseArray, Pose, PoseStamped


class RosSelfTest(Node):
    def __init__(self):
        super().__init__("tracker_ros_self_test")
        self.pose_array_pub = self.create_publisher(PoseArray, "/robots/pos", 10)
        self.robot_1_pub = self.create_publisher(PoseStamped, "/robot_1/pose", 10)

    def publish_test_messages(self):
        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = "map"

        pose = Pose()
        pose.position.x = 123.0
        pose.position.y = 45.0
        pose.position.z = 0.0
        pose.orientation.w = 1.0
        pose_array.poses.append(pose)

        pose_stamped = PoseStamped()
        pose_stamped.header = pose_array.header
        pose_stamped.pose = pose

        self.pose_array_pub.publish(pose_array)
        self.robot_1_pub.publish(pose_stamped)

        self.get_logger().info("Published test pose to /robots/pos and /robot_1/pose")


def main():
    rclpy.init()
    node = RosSelfTest()

    try:
        for _ in range(20):
            node.publish_test_messages()
            rclpy.spin_once(node, timeout_sec=0.1)
            time.sleep(0.5)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
