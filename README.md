# Tracking Module

AprilTag-based robot localization for the Jachtseizoen robot swarm project.

The tracker runs as a containerized ROS 2 Humble node. It connects to the server through WireGuard and publishes robot positions over the CycloneDDS ROS 2 network.

## Published topics

- `/cam/pos` — `geometry_msgs/msg/PoseArray`
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
CAMERA_PROFILE=sotp_cam
REFERENCE_TAG_0=0
REFERENCE_TAG_1=1
TARGET_TAGS=2:robot_2,3:robot_3,4:robot_4,5:robot_5
T0X=-0.3
T0Y=-0.3
T1X=10.3
T1Y=10.3
TAG_SIZE=0.025
OUTPUT_TOPIC=/cam/pos
```

Adjust these in `compose.remote.yaml` for the physical setup.

## Calibration

Example:

```bash
python src/calibrate/calibrate_camera.py -r 6 -c 9 -s 23 -i 5
```
