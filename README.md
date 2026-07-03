# Tracking Module

AprilTag-based localization service for the Jachtseizoen robot swarm. The tracker reads an overhead camera feed, calibrates the playing field from reference tags, detects robot tags, and publishes physical robot positions into the ROS 2 system.

## Features

- Overhead camera tracking with AprilTags.
- Field calibration using two reference tags.
- Configurable target-tag to robot-ID mapping.
- Aggregate PoseArray output for all detected physical robots.
- Optional per-robot PoseStamped output.
- Optional browser-accessible debug stream for camera and tag verification.
- Docker deployment with WireGuard and CycloneDDS.

## Repository contents

| Path | Purpose |
|---|---|
| `src/main.py` | Main tracking pipeline. |
| `src/locator.py` | AprilTag detection and pose estimation. |
| `src/ros_worker.py` | ROS 2 publishers for tracked robot poses. |
| `src/stream_server.py` | Optional web stream server. |
| `src/threaded_cam.py` | Camera capture wrapper. |
| `src/defines.py` | Camera calibration profiles. |
| `src/calibrate/calibrate_camera.py` | Camera calibration utility. |
| `compose.remote.yaml` | Final remote tracker deployment. |
| `compose.test.yaml` | Local test deployment. |
| `cyclonedds/client.xml` | CycloneDDS client configuration. |

## Getting started

### Requirements

- Docker and Docker Compose
- Linux machine with the overhead camera connected
- WireGuard peer configuration for the Jacht server
- Access to the project ROS 2 network

### Install

1. Clone the repository.
2. Place the WireGuard peer configuration in the expected path:

```bash
mkdir -p wireguard-client/wg_confs
cp wg0.conf wireguard-client/wg_confs/wg0.conf
```

3. Confirm that the camera device in `compose.remote.yaml` matches the host machine. The current final setup maps host `/dev/video5` to container `/dev/video0`.
4. Start the tracker:

```bash
docker compose -f compose.remote.yaml up -d --build
```

5. Follow the logs:

```bash
docker compose -f compose.remote.yaml logs -f tracker
```

6. Open the debug stream when the tracker is started with `--show`:

```text
http://<tracker-host>:8080
```

When using the server-side proxy, use:

```text
http://<server-host>:18080
```

Stop the tracker:

```bash
docker compose -f compose.remote.yaml down --remove-orphans
```

## Configuration

The final remote compose file uses the following values.

| Variable / option | Current value | Effect |
|---|---:|---|
| `ROS_DOMAIN_ID` | `0` | ROS 2 domain ID. |
| `ROS_LOCALHOST_ONLY` | `0` | Allows ROS traffic outside localhost. |
| `RMW_IMPLEMENTATION` | `rmw_cyclonedds_cpp` | DDS implementation. |
| `CYCLONEDDS_URI` | `file:///cyclonedds/client.xml` | CycloneDDS config file. |
| `CAMERA_INDEX` | `0` | Camera index inside the container. |
| Host camera device | `/dev/video5` | Camera device on the tracker host. |
| Container camera device | `/dev/video0` | Camera device visible inside the tracker container. |
| `CAMERA_PROFILE` | `sotp_cam` | Camera calibration profile from `src/defines.py`. |
| `REFERENCE_TAG_0` | `0` | First reference tag. |
| `REFERENCE_TAG_1` | `1` | Second reference tag. |
| `TARGET_TAGS` | `2:robot_2,3:robot_3,4:robot_4,5:robot_5` | Physical tag-to-robot mapping. |
| `T0X`, `T0Y` | `-0.3`, `-0.3` | Field coordinate for reference tag 0. |
| `T1X`, `T1Y` | `10.3`, `10.3` | Field coordinate for reference tag 1. |
| `TAG_SIZE` | `0.025` source default | AprilTag size in metres. |
| `OUTPUT_TOPIC` | `/cam/pos` | Aggregate physical robot pose output. |
| `--publish-individual` | disabled | Enables one PoseStamped topic per robot when explicitly passed. |
| `--show` | enabled in `compose.remote.yaml` | Starts the debug stream and visual overlay pipeline. |

## Actions

| Action | Command / method |
|---|---|
| Start tracker | `docker compose -f compose.remote.yaml up -d --build` |
| Stop tracker | `docker compose -f compose.remote.yaml down --remove-orphans` |
| View logs | `docker compose -f compose.remote.yaml logs -f tracker` |
| View debug stream | Open `http://<tracker-host>:8080` or server proxy `http://<server-host>:18080` |
| Check output | `ros2 topic echo /cam/pos` from a ROS 2 shell on the project network |

## Calibration

### Field calibration

Field calibration is automatic at startup:

1. Place reference tag `0` and reference tag `1` at the configured field reference positions.
2. Start the tracker.
3. Keep both reference tags visible until calibration completes.
4. After calibration, place the robot tags in view. The tracker publishes mapped robot poses when target tags are detected.

The tracker gathers 100 valid calibration frames before locking the field transform.

### Camera calibration utility

Use the camera calibration script when a new camera profile is needed:

```bash
python src/calibrate/calibrate_camera.py -r 6 -c 9 -s 23 -i 5
```

*Note: The command above is for a 7 rows, 10 columns, 23 mm checkerboard pattern, calibrating camera with id 5 (`/dev/video5`).*

The generated calibration data must be added from `camera_calib.json` to `src/defines.py` and selected with `CAMERA_PROFILE`.

## Connections

| Direction | Interface | Purpose |
|---|---|---|
| Incoming | USB/V4L2 camera device | Overhead image input. |
| Incoming | WireGuard client | Joins the server ROS 2 network. |
| Outgoing | `/cam/pos` (`geometry_msgs/msg/PoseArray`) | Aggregate physical robot positions. |
| Optional outgoing | `/<robot_id>/pose` (`geometry_msgs/msg/PoseStamped`) | Individual debug pose output when enabled. |
| Outgoing | HTTP `8080` | Debug stream when `--show` is active. |
| Server-facing | Tracker UI proxy on server port `18080` | Forwards the tracker debug stream through the server. |
