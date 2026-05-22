import argparse
import json
import os
import time

import apriltag
import cv2
import numpy as np

import defines


def parse_args():
    parser = argparse.ArgumentParser(description="Headless AprilTag tracker")
    parser.add_argument("--camera-index", type=int, default=int(os.getenv("CAMERA_INDEX", "0")))
    parser.add_argument("--show", action="store_true", help="Show OpenCV debug window")
    parser.add_argument("--scale-x", type=float, default=float(os.getenv("SCALE_X", "200")))
    parser.add_argument("--scale-y", type=float, default=float(os.getenv("SCALE_Y", "200")))
    parser.add_argument("--reference-tag-0", type=int, default=int(os.getenv("REFERENCE_TAG_0", "0")))
    parser.add_argument("--reference-tag-1", type=int, default=int(os.getenv("REFERENCE_TAG_1", "1")))
    parser.add_argument("--target-tags", default=os.getenv("TARGET_TAGS", "2,3"))
    parser.add_argument("--camera-profile", default=os.getenv("CAMERA_PROFILE", "microsoft_cam"))
    return parser.parse_args()


def get_camera_calibration(profile_name):
    if not hasattr(defines, profile_name):
        raise ValueError(f"Unknown camera profile: {profile_name}")
    return getattr(defines, profile_name)


def get_center(det):
    return np.array(det["center"], dtype=np.float32)


def get_corners(det):
    return np.array(det["lb-rb-rt-lt"], dtype=np.int32)


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
    homogenous = np.array([pixel[0], pixel[1], 1.0], dtype=np.float32)
    mapped = np.dot(matrix, homogenous)
    return float(mapped[0]), float(mapped[1])


def main():
    args = parse_args()

    target_tags = [int(x.strip()) for x in args.target_tags.split(",") if x.strip()]
    cam_cal = get_camera_calibration(args.camera_profile)

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
        raise RuntimeError(f"Could not open camera index {args.camera_index}")

    print(json.dumps({
        "event": "tracker_started",
        "camera_index": args.camera_index,
        "camera_profile": args.camera_profile,
        "reference_tags": [args.reference_tag_0, args.reference_tag_1],
        "target_tags": target_tags,
        "scale": [args.scale_x, args.scale_y],
        "show": args.show,
    }), flush=True)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(json.dumps({"event": "camera_read_failed"}), flush=True)
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
                    corners = get_corners(det)
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

            robots = []

            if transform is not None:
                for target_id in target_tags:
                    if target_id not in frame_tags:
                        continue

                    x, y = apply_transform(transform, frame_tags[target_id])
                    robots.append({
                        "tag_id": target_id,
                        "x": round(x, 3),
                        "y": round(y, 3),
                    })

                    if args.show:
                        pixel = frame_tags[target_id].astype(int)
                        cv2.putText(
                            frame,
                            f"Mapped: ({int(x)}, {int(y)})",
                            (int(pixel[0]), int(pixel[1]) + 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 255),
                            1,
                        )

            output = {
                "event": "frame",
                "visible_tags": sorted(list(frame_tags.keys())),
                "reference_visible": transform is not None,
                "robots": robots,
            }

            print(json.dumps(output), flush=True)

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

                cv2.imshow("AprilTag Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            time.sleep(0.05)

    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
