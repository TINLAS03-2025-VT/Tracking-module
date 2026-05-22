from ros:latest as ros-server

env DEBIAN_FRONTEND=noninteractive

run apt-get update && apt-get install -y \
	ros-${ROS_DISTRO}-demo-nodes-cpp \
	ros-${ROS_DISTRO}-demo-nodes-py \
	ros-${ROS_DISTRO}-rosbridge-server && \
    rm -rf /var/lib/apt/lists/*

expose 9090/tcp
expose 7400/udp
expose 7401/udp

run echo "if [ -f /opt/ros/jazzy/setup.bash ]; then\n	source /opt/ros/jazzy/setup.bash\nfi" >> /etc/bash.bashrc

cmd ["bash", "-c", "ros2 launch rosbridge_server rosbridge_websocket_launch.xml address:=0.0.0.0", "&"]
