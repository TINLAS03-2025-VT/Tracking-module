# Tracking Module

AprilTag-based robot localization for the Jachtseizoen robot swarm project.

The tracker runs as a containerized ROS 2 Humble node. It connects to the server through WireGuard and publishes robot positions over the CycloneDDS ROS 2 network.

## Published topics

- `/robots/pos` — `geometry_msgs/msg/PoseArray`
- `/<robot_id>/pose` — `geometry_msgs/msg/PoseStamped`, for example `/robot_1/pose`

## Requirements

- Docker + Docker Compose
- A camera connected to the tracker machine
- A WireGuard peer config from the server maintainer

## WireGuard config

After cloning the repo, save your WireGuard peer config here:

```text
wireguard-client/wg_confs/wg0.conf
```

Example:

```bash
mkdir -p wireguard-client/wg_confs
nano wireguard-client/wg_confs/wg0.conf
```

Do not commit `wireguard-client/`. It contains private keys and should stay ignored by Git.

## No-camera ROS test

Use this first to test WireGuard + ROS 2 without a camera:

```bash
docker compose -f compose.remote.test.yaml pull
docker compose -f compose.remote.test.yaml up
```

Expected output:

```text
tracker-test contained wg0 ready
Published test pose to /robots/pos and /robot_1/pose
```

On the server, check:

```bash
ros2 topic echo /robots/pos geometry_msgs/msg/PoseArray
```

Stop the test:

```bash
docker compose -f compose.remote.test.yaml down --remove-orphans
```

## Run the real tracker

By default the tracker expects the camera at `/dev/video0`.

```bash
docker compose -f compose.remote.yaml pull
docker compose -f compose.remote.yaml up -d
docker logs -f tracking-module
```

Stop it:

```bash
docker compose -f compose.remote.yaml down --remove-orphans
```

## Camera/config environment variables

The remote Compose file sets:

```text
CAMERA_INDEX=0
CAMERA_PROFILE=microsoft_cam
REFERENCE_TAG_0=0
REFERENCE_TAG_1=1
TARGET_TAGS=2:robot_1,3:robot_2
SCALE_X=200
SCALE_Y=200
OUTPUT_TOPIC=/robots/pos
PUBLISH_INDIVIDUAL_POSES=true
```

Adjust these in `compose.remote.yaml` for the physical setup.

## Local container self-test

This does not require ROS networking or a camera:

```bash
docker compose -f compose.test.yaml up --build
```

## Calibration

Example:

```bash
python src/calibrate_camera.py -r 6 -c 9 -s 23 -i 5
```
