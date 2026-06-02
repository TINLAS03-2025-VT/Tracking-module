import argparse
import json
import os
import time
import math

import apriltag
import cv2
import numpy as np

import defines


def parse_args():
    valid_profiles = [
        attr for attr in dir(defines) 
        if not attr.startswith("__") and not hasattr(getattr(defines, attr), "__call__")
    ]
    profiles_str = ", ".join(valid_profiles) if valid_profiles else "none found"

    parser = argparse.ArgumentParser(description="Headless AprilTag tracker")
    parser.add_argument("-c", "--camera-index", type=int, default=int(os.getenv("CAMERA_INDEX", "0")), help="index of the camera to use (default: 0)")
    parser.add_argument("-s", "--show", action="store_true", help="show OpenCV debug window")
    parser.add_argument("-t", "--tag-size", type=float, default=float(os.getenv("TAG_SIZE", "0.025")), help="inner dimensions of the tags in meters (default: 0.025)")
    parser.add_argument("-p", "--camera-profile", default=os.getenv("CAMERA_PROFILE", "microsoft_cam"), choices=valid_profiles, help=f"camera calibration profile from defines.py. Available options: {profiles_str}")
    parser.add_argument("-x", "--scale-x", type=float, default=float(os.getenv("SCALE_X", "200")), help="x-axis scale to map the locations to (default: 200.0)")
    parser.add_argument("-y", "--scale-y", type=float, default=float(os.getenv("SCALE_Y", "200")), help="y-axis scale to map the locations to (default: 200.0)")
    parser.add_argument("-0", "--reference-tag-0", type=int, default=int(os.getenv("REFERENCE_TAG_0", "0")), help="tag ID for reference tag at location {x: 0, y: 0} (default: 0)")
    parser.add_argument("-1", "--reference-tag-1", type=int, default=int(os.getenv("REFERENCE_TAG_1", "1")), help="tag ID for reference tag at location {x: SCALE_X, y: SCALE_Y} (default: 1)")
    parser.add_argument("-a", "--alpha", type=float, default=float(os.getenv("TRACKER_ALPHA", "0.3")), help="weight alpha for exponential moving average (default: 0.3)")
    parser.add_argument("--disable-ocl", action="store_true", help="force disable OpenCL acceleration")
    return parser.parse_args()


def get_camera_calibration(profile_name):
    if not hasattr(defines, profile_name):
        raise ValueError(f"Unknown camera profile: {profile_name}")
    return getattr(defines, profile_name)


def make_4x4_matrix(R, t):
    # Combines a 3x3 rotation matrix and a 3x1 translation vector into a 4x4 homogeneous matrix.
    T = np.eye(4, dtype=np.float32)
    T[0:3, 0:3] = R
    T[0:3, 3] = t.flatten()
    return T

def main():
    args = parse_args()
    cam_cal = get_camera_calibration(args.camera_profile)
    fx, fy, cx, cy = cam_cal["fx"], cam_cal["fy"], cam_cal["cx"], cam_cal["cy"]

    if args.disable_ocl:
        cv2.ocl.setUseOpenCL(False)
    else:
        cv2.ocl.setUseOpenCL(True)
    ocl_enabled = cv2.ocl.useOpenCL()

    decimate_factor = 2.0

    fx_det, fy_det = fx / decimate_factor, fy / decimate_factor
    cx_det, cy_det = cx / decimate_factor, cy / decimate_factor

    detector = apriltag.apriltag(
    	family='tagStandard41h12',  # Tag family
	    threads=4,                  # Number of threads
	    maxhamming=1,               # Maximum hamming distance for error correction
	    decimate=1.0,               # Image downsampling factor
	    blur=0.0,                   # Gaussian blur sigma
	    refine_edges=True,          # Refine quad edges
	    debug=False                 # Debug mode
    )

    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera_index}")

    tracked_robots = {}
    ref_tags = {args.reference_tag_0, args.reference_tag_1}

    map_calibrated = False
    R_ref0_to_stable = None
    scale_factor = 1.0
    theta_rad = 0.0

    print(json.dumps({
        "event": "tracker_started",
        "camera_index": args.camera_index,
        "tag_size": args.tag_size,
        "camera_profile": args.camera_profile,
        "reference_tags": list(ref_tags),
        "scale": [args.scale_x, args.scale_y],
        "show": args.show,
        "alpha": args.alpha,
        "opencl_accelerated": ocl_enabled
    }), flush=True)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(json.dumps({"event": "camera_read_failed"}), flush=True)
                time.sleep(0.2)
                continue

            if ocl_enabled:
                umat_frame = cv2.UMat(frame)
                umat_gray = cv2.cvtColor(umat_frame, cv2.COLOR_BGR2GRAY)
                umat_gray = cv2.GaussianBlur(umat_gray, (3, 3), 0)

                if decimate_factor > 1.0:
                    width = int(frame.shape[1] / decimate_factor)
                    height = int(frame.shape[0] / decimate_factor)
                    umat_gray = cv2.resize(umat_gray, (width, height), interpolation=cv2.INTER_LINEAR)

                gray_for_detector = umat_gray.get()
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if decimate_factor > 1.0:
                    width = int(frame.shape[1] / decimate_factor)
                    height = int(frame.shape[0] / decimate_factor)
                    gray = cv2.resize(gray, (width, height), interpolation=cv2.INTER_LINEAR)
                gray_for_detector = gray

            detections = detector.detect(gray_for_detector)

            frame_poses = {}
            frame_centers = {}

            for det in detections:
                tag_id = int(det["id"])
                center = np.array(det["center"], dtype=np.float32) * decimate_factor
                frame_centers[tag_id] = center

                pose = detector.estimate_tag_pose(det, args.tag_size, fx_det, fy_det, cx_det, cy_det)
                T = make_4x4_matrix(pose['R'], pose['t'])
                T[0:3, 3] *= decimate_factor
                frame_poses[tag_id] = T

                if args.show:
                    corners = (np.array(det["lb-rb-rt-lt"], dtype=np.int32) * decimate_factor).astype(np.int32)
                    center_int = tuple(center.astype(int))
                    cv2.polylines(frame, [corners], True, (0,255,0), 2)
                    cv2.circle(frame, center_int, 5, (0, 0, 255), -1)
                    cv2.putText(frame, str(tag_id), center_int, cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            if args.reference_tag_0 in frame_poses and args.reference_tag_1 in frame_poses:
                T_cam_ref0 = frame_poses[args.reference_tag_0]
                T_cam_ref1 = frame_poses[args.reference_tag_1]
                T_ref0_cam = np.linalg.inv(T_cam_ref0)

                p1_in_ref0 = T_ref0_cam @ np.append(T_cam_ref1[0:3, 3], 1.0)
                v_ref0 = p1_in_ref0[0:3]
                d_phys = np.linalg.norm(v_ref0)

                if d_phys > 0.001:
                    n_ref0 = np.array([0.0, 0.0, 1.0], dtype=np.float32)
                    u_y_ref0 = np.cross(n_ref0, v_ref0)
                    u_y_norm = np.linalg.norm(u_y_ref0)

                    if u_y_norm > 0.001:
                        u_y_ref0 /= u_y_norm
                        u_z_ref0 = np.cross(v_ref0, u_y_ref0)
                        u_z_ref0 /= np.linalg.norm(u_z_ref0)
                        u_x_ref0 = np.cross(u_y_ref0, u_z_ref0)
                        u_x_ref0 /= np.linalg.norm(u_x_ref0)

                        R_ref0_to_stable = np.stack([u_x_ref0, u_y_ref0, u_z_ref0], axis=1)

                        d_map = math.hypot(args.scale_x, args.scale_y)
                        scale_factor = d_map / d_phys
                        theta_rad = math.atan2(args.scale_y, args.scale_x)
                        map_calibrated = True

            if map_calibrated and args.reference_tag_0 in frame_poses:
                T_cam_ref0 = frame_poses[args.reference_tag_0]
                T_ref0_cam = np.linalg.inv(T_cam_ref0)

                cos_t = math.cos(theta_rad)
                sin_t = math.sin(theta_rad)

                if args.show:
                    map_corners = [(0, 0), (args.scale_x, 0), (args.scale_x, args.scale_y), (0, args.scale_y)]
                    field_corners_3d = []
                    inv_s = 1.0 / scale_factor

                    for mx, my in map_corners:
                        xf = inv_s * (mx * cos_t + my * sin_t)
                        yf = inv_s * (-mx * sin_t + my * cos_t)
                        v_stable = np.array([xf, yf, 0.0], dtype=np.float32)

                        v_ref0 = R_ref0_to_stable @ v_stable
                        p_cam = T_cam_ref0 @ np.append(v_ref0, 1.0)
                        field_corners_3d.append(p_cam[0:3])

                    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
                    dist_coeffs = np.zeros(4, dtype=np.float32)
                    img_pts, _ = cv2.projectPoints(np.array(field_corners_3d, dtype=np.float32), np.zeros(3), np.zeros(3), camera_matrix, dist_coeffs)
                    img_pts = img_pts.reshape(-1, 2).astype(np.int32)

                    cv2.polylines(frame, [img_pts], True, (255, 0, 255), 2)
                    cv2.putText(frame, "Field Plane Grid", tuple(img_pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

                for tag_id, T_cam_target in frame_poses.items():
                    if tag_id in ref_tags:
                        continue

                    p_target_in_ref0 = T_ref0_cam @ np.append(T_cam_target[0:3, 3], 1.0)
                    v_target_stable = R_ref0_to_stable.T @ p_target_in_ref0[0:3]
                    x_floor, y_floor = v_target_stable[0], v_target_stable[1]

                    raw_x = scale_factor * (x_floor * cos_t - y_floor * sin_t)
                    raw_y = scale_factor * (x_floor * sin_t + y_floor * cos_t)

                    T_ref0_target = T_ref0_cam @ T_cam_target
                    R_stable_target = R_ref0_to_stable.T @ T_ref0_target[0:3, 0:3]

                    R_z = np.array([
                        [cos_t, -sin_t, 0],
                        [sin_t, cos_t, 0],
                        [0, 0, 1]
                    ], dtype=np.float32)
                    R_map_target = R_z.T @ R_stable_target
                    map_angle_deg = math.degrees(math.atan2(R_map_target[1, 0], R_map_target[0, 0])) % 360.0

                    if tag_id in tracked_robots:
                        old_x = tracked_robots[tag_id]["x"]
                        old_y = tracked_robots[tag_id]["y"]
                        old_rot = tracked_robots[tag_id]["rotation"]

                        filtered_x = (args.alpha * raw_x) + ((1 - args.alpha) * old_x)
                        filtered_y = (args.alpha * raw_y) + ((1 - args.alpha) * old_y)

                        sin_sum = args.alpha * math.sin(math.radians(map_angle_deg)) + (1 - args.alpha) * math.sin(math.radians(old_rot))
                        cos_sum = args.alpha * math.cos(math.radians(map_angle_deg)) + (1 - args.alpha) * math.cos(math.radians(old_rot))
                        filtered_rot = math.degrees(math.atan2(sin_sum, cos_sum)) % 360
                    else:
                        filtered_x = raw_x
                        filtered_y = raw_y
                        filtered_rot = map_angle_deg

                    tracked_robots[tag_id] = {
                        "x": round(filtered_x, 3),
                        "y": round(filtered_y, 3),
                        "rotation": round(filtered_rot, 2),
                    }

            robots_output = []
            for tag_id, pos in tracked_robots.items():
                robots_output.append({
                    "tag_id": tag_id,
                    "x": pos["x"],
                    "y": pos["y"],
                    "rotation_deg": pos["rotation"],
                    "currently_visible": tag_id in frame_poses,
                })

                if args.show and tag_id in frame_centers:
                    pixel = frame_centers[tag_id].astype(int)
                    label = f"X:{pos['x']:.2f} Y:{pos['y']:.2f} R:{int(pos['rotation'])}deg"
                    cv2.putText(frame, label, (pixel[0], pixel[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 122, 0), 2)

            output = {
                "event": "frame",
                "visible_tags": sorted(list(frame_centers.keys())),
                "reference_visible": (args.reference_tag_0 in frame_poses and args.reference_tag_1 in frame_poses),
                "map_calibrated": map_calibrated,
                "robots": robots_output,
            }
            print(json.dumps(output), flush=True)

            if args.show:
                cv2.imshow("AprilTag 3D Pose Tracker", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            time.sleep(0.01)

    finally:
        cap.release()
        if args.show:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
