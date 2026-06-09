import math
import numpy as np
import apriltag

class AprilTagTrackerEngine:
    def __init__(self, tag_size: float, alpha: float, scale_x: float, scale_y: float,
                 ref_tag_0: int, ref_tag_1: int, fx: float, fy: float, cx: float, cy: float):
        self.tag_size = tag_size
        self.alpha = alpha
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.ref_tag_0 = ref_tag_0
        self.ref_tag_1 = ref_tag_1

        # Camera Intrinsics
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
        self.dist_coeffs = np.zeros(4, dtype=np.float32)

        # Detector Engine
        self.detector = apriltag.apriltag(family="tagStandard41h12", threads=8, refine_edges=True, debug=False)

        # Calibration State
        self.map_calibrated = False
        self.R_ref0_to_stable = None
        self.scale_factor = 1.0
        self.theta_rad = 0.0

        self.smoothed_R_ref0 = None
        self.smoothed_scale_factor = 1.0
        self.smoothed_theta_rad = 0.0
        self.calibration_frames_tracked = 0
        self.CALIBRATION_LOCK_THRESHOLD = 50

        # Robot Filtering State
        self.tracked_robots = {}
        self.angle_buffers = {}

    @staticmethod
    def make_4x4_matrix(R, t):
        T = np.eye(4, dtype=np.float32)
        T[0:3, 0:3] = R
        T[0:3, 3] = t.flatten()
        return T

    def process_frame(self, gray_img):
        """Detects tags and extracts raw 3D transformations."""
        detections = self.detector.detect(gray_img)
        frame_poses = {}
        frame_centers = {}

        for det in detections:
            tag_id = int(det["id"])
            frame_centers[tag_id] = np.array(det["center"], dtype=np.float32)
            pose = self.detector.estimate_tag_pose(det, self.tag_size, self.fx, self.fy, self.cx, self.cy)
            frame_poses[tag_id] = self.make_4x4_matrix(pose['R'], pose['t'])

        return detections, frame_poses, frame_centers

    def compute_calibration(self, frame_poses, logger=None):
        """Calculates and locks the transformation plane using reference tags."""
        if self.ref_tag_0 not in frame_poses or self.ref_tag_1 not in frame_poses:
            return

        if self.calibration_frames_tracked >= self.CALIBRATION_LOCK_THRESHOLD and self.CALIBRATION_LOCK_THRESHOLD != 0:
            return

        T_cam_ref0 = frame_poses[self.ref_tag_0]
        T_cam_ref1 = frame_poses[self.ref_tag_1]
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

                raw_R = np.stack([u_x_ref0, u_y_ref0, u_z_ref0], axis=1)
                raw_scale = math.hypot(self.scale_x, self.scale_y) / d_phys
                raw_theta = math.atan2(self.scale_y, self.scale_x)

                cal_alpha = 0.1
                if self.smoothed_R_ref0 is None:
                    self.smoothed_R_ref0 = raw_R
                    self.smoothed_scale_factor = raw_scale
                    self.smoothed_theta_rad = raw_theta
                else:
                    self.smoothed_R_ref0 = (cal_alpha * raw_R) + ((1.0 - cal_alpha) * self.smoothed_R_ref0)
                    self.smoothed_scale_factor = (cal_alpha * raw_scale) + ((1.0 - cal_alpha) * self.smoothed_scale_factor)
                    self.smoothed_theta_rad = self.smoothed_theta_rad + cal_alpha * math.atan2(
                        math.sin(raw_theta - self.smoothed_theta_rad),
                        math.cos(raw_theta - self.smoothed_theta_rad)
                    )

                U, _, Vt = np.linalg.svd(self.smoothed_R_ref0)
                self.R_ref0_to_stable = U @ Vt
                self.scale_factor = self.smoothed_scale_factor
                self.theta_rad = self.smoothed_theta_rad

                self.calibration_frames_tracked += 1
                self.map_calibrated = True

                if self.calibration_frames_tracked == self.CALIBRATION_LOCK_THRESHOLD and logger:
                    logger.info("🔑 Calibration matrix securely LOCKED.")

    def get_field_grid_points(self, frame_poses):
        """Generates pixel array projections for rendering the ground field plane grid."""
        if not self.map_calibrated or self.ref_tag_0 not in frame_poses:
            return None

        import cv2
        T_cam_ref0 = frame_poses[self.ref_tag_0]
        cos_t, sin_t = math.cos(self.theta_rad), math.sin(self.theta_rad)
        map_corners = [(0, 0), (self.scale_x, 0), (self.scale_x, self.scale_y), (0, self.scale_y)]
        field_corners_3d = []
        inv_s = 1.0 / self.scale_factor

        for mx, my in map_corners:
            xf = inv_s * (mx * cos_t + my * sin_t)
            yf = inv_s * (-mx * sin_t + my * cos_t)
            v_stable = np.array([xf, yf, 0.0], dtype=np.float32)
            v_ref0 = self.R_ref0_to_stable @ v_stable
            p_cam = T_cam_ref0 @ np.append(v_ref0, 1.0)
            field_corners_3d.append(p_cam[0:3])

        img_pts, _ = cv2.projectPoints(
            np.array(field_corners_3d, dtype=np.float32),
            np.zeros(3), np.zeros(3),
            self.camera_matrix, self.dist_coeffs
        )
        return img_pts.reshape(-1, 2).astype(np.int32)

    def localize_target(self, target_id, frame_poses):
        """Calculates 2D space locations and filtered heading orientation angles."""
        if not self.map_calibrated or self.ref_tag_0 not in frame_poses or target_id not in frame_poses:
            return None

        T_cam_ref0 = frame_poses[self.ref_tag_0]
        T_ref0_cam = np.linalg.inv(T_cam_ref0)
        cos_t, sin_t = math.cos(self.theta_rad), math.sin(self.theta_rad)

        T_cam_target = frame_poses[target_id]
        p_target_in_ref0 = T_ref0_cam @ np.append(T_cam_target[0:3, 3], 1.0)
        v_target_stable = self.R_ref0_to_stable.T @ p_target_in_ref0[0:3]
        x_floor, y_floor = v_target_stable[0], v_target_stable[1]

        raw_x = self.scale_factor * (x_floor * cos_t - y_floor * sin_t)
        raw_y = self.scale_factor * (x_floor * sin_t + y_floor * cos_t)

        T_ref0_target = T_ref0_cam @ T_cam_target
        R_stable_target = self.R_ref0_to_stable.T @ T_ref0_target[0:3, 0:3]

        R_z = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]], dtype=np.float32)
        R_map_target = R_z.T @ R_stable_target
        map_angle_deg = math.degrees(math.atan2(R_map_target[1, 0], R_map_target[0, 0])) % 360.0

        # Apply Exponential Smoothing Filters
        if target_id in self.tracked_robots:
            old_x = self.tracked_robots[target_id]["x"]
            old_y = self.tracked_robots[target_id]["y"]
            filtered_x = (self.alpha * raw_x) + ((1 - self.alpha) * old_x)
            filtered_y = (self.alpha * raw_y) + ((1 - self.alpha) * old_y)

            current_sin = math.sin(math.radians(map_angle_deg))
            current_cos = math.cos(math.radians(map_angle_deg))
            self.angle_buffers[target_id]["sin"] = (self.alpha * current_sin) + ((1 - self.alpha) * self.angle_buffers[target_id]["sin"])
            self.angle_buffers[target_id]["cos"] = (self.alpha * current_cos) + ((1 - self.alpha) * self.angle_buffers[target_id]["cos"])
            filtered_rot = math.degrees(math.atan2(self.angle_buffers[target_id]["sin"], self.angle_buffers[target_id]["cos"])) % 360
        else:
            filtered_x, filtered_y, filtered_rot = raw_x, raw_y, map_angle_deg
            self.angle_buffers[target_id] = {
                "sin": math.sin(math.radians(map_angle_deg)),
                "cos": math.cos(math.radians(map_angle_deg))
            }

        self.tracked_robots[target_id] = {
            "x": round(filtered_x, 3), "y": round(filtered_y, 3), "rotation": round(filtered_rot, 2)
        }

        return filtered_x, filtered_y, filtered_rot