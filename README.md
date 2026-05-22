# Tracking Module

AprilTag tracking module for the TINLAS03 Jachtseizoen robot project.

Current milestone:

- Docker container builds
- Python starts
- OpenCV imports
- AprilTag library imports
- Detector can run on a dummy image
- No ROS yet

## Self-test without camera

```bash
docker compose -f compose.test.yaml build --no-cache
docker compose -f compose.test.yaml run --rm tracker-test

cat > README.md <<'EOF'
# Tracking Module

AprilTag tracking module for the TINLAS03 Jachtseizoen robot project.

Current milestone:

- Docker container builds
- Python starts
- OpenCV imports
- AprilTag library imports
- Detector can run on a dummy image
- No ROS yet

## Self-test without camera

```bash
docker compose -f compose.test.yaml build --no-cache
docker compose -f compose.test.yaml run --rm tracker-test



