from __future__ import print_function
import numpy as np
import cv2
import argparse
import time
import json

def board_moved_significantly(last_corners, current_corners, threshold=40.0):
    """Ensures we only capture frames when the board actually moves."""
    if last_corners is None:
        return True

    # Calculate the mean center of the chessboard for both frames
    last_center = np.mean(last_corners, axis=0)[0]
    curr_center = np.mean(current_corners, axis=0)[0]

    # Calculate Euclidean distance between the centers
    dist = np.linalg.norm(curr_center - last_center)
    return dist > threshold

def main():

    parser = argparse.ArgumentParser(
        description='Calibrate camera intrinsics using OpenCV via live video stream')
    parser.add_argument('-i', '--camera-index', type=int, default=0,
                        help='Index of the camera to use (default: 0)')
    parser.add_argument('-r', '--rows', metavar='N', type=int, required=True,
                        help='# of chessboard corners in vertical direction')
    parser.add_argument('-c', '--cols', metavar='N', type=int, required=True,
                        help='# of chessboard corners in horizontal direction')
    parser.add_argument('-s', '--size', metavar='NUM', type=float, default=1.0,
                        help='chessboard square size in user-chosen units')
    parser.add_argument('-t', '--time', metavar='N', type=int, default=30,
                        help='time the calibration runs for')
    parser.add_argument('-o', '--output', type=str, default='camera_calib.json',
                        help='Output JSON file to save calibration data')

    options = parser.parse_args()

    patternsize = (options.cols, options.rows)
    sz = options.size

    term_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    x = np.arange(patternsize[0]) * sz
    y = np.arange(patternsize[1]) * sz
    xgrid, ygrid = np.meshgrid(x, y)
    zgrid = np.zeros_like(xgrid)
    opoints_single = np.dstack((xgrid, ygrid, zgrid)).reshape((-1, 3)).astype(np.float32)

    imagesize = None
    win = 'Calibrate - Live Stream (Vary distance and angles!)'
    cv2.namedWindow(win)

    ipoints = []
    opoints = []
    last_saved_corners = None

    cap = cv2.VideoCapture(options.camera_index)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Starting video capture. Move the chessboard around in front of the camera.")
    print(f"Capturing will run for {options.time} seconds. Press 'q' to stop early.")

    duration = options.time
    start_time = time.time()
    last_sample_time = 0

    while True:
        elapsed_time = time.time()- start_time
        remaining_time = max(0, int(duration - elapsed_time))

        if elapsed_time > duration:
            print("\nFinished gathering calibration data.")
            break

        ret, rgb = cap.read()
        if not ret:
            print("Failed to get frame.")
            time.sleep(0.005)
            break

        if imagesize is None:
            imagesize = (rgb.shape[1], rgb.shape[0])

        gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY) if len(rgb.shape) == 3 else rgb

        retval, corners = cv2.findChessboardCorners(gray, patternsize, None)
        display = rgb.copy()

        if retval:
            cv2.drawChessboardCorners(display, patternsize, corners, retval)

            current_time = time.time()
            if current_time - last_sample_time > 0.5:
                if board_moved_significantly(last_saved_corners, corners):
                    corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term_criteria)

                    ipoints.append(corners2)
                    opoints.append(opoints_single)
                    last_saved_corners = corners2
                    last_sample_time = current_time

                    cv2.rectangle(display, (0,0), (display.shape[1], display.shape[0]), (0, 255, 0), 10)
                    print("Saved screenshot {} | Remaining: {}s".format(len(ipoints), remaining_time))
                else:
                    cv2.putText(display, "MOVE BOARD MORE", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
        else:
            print("Failed to find chessboard.")
            time.sleep(0.005)

        cv2.putText(display, "Time Left: {}s".format(remaining_time), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(display, "Snapshots: {}".format(len(ipoints)), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow(win, display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nStopped early by user. Starting calibration...")
            break

    cv2.destroyWindow(win)

    if len(ipoints) < 10:
        print(f"Error: Only {len(ipoints)} patterns detected. Minimum 10 needed for reliable calibration.")
        cap.release()
        return

    print(f"\nCalibrating camera matrix based on {len(ipoints)} diverse frames...")

    rms, K, dcoeffs, rvecs, tvecs = cv2.calibrateCamera(
        opoints, ipoints, imagesize, None, None, flags=0
    )

    print("\n" + "="*40)
    print(f"RMS REPROJECTION ERROR: {rms:.4f} pixels")
    if rms < 0.5:
        print(" -> Excellent calibration!")
    elif rms < 1.0:
        print(" -> Good calibration.")
    else:
        print(" -> Poor calibration. Consider re-running and keeping the board flatter.")
    print("="*40 + "\n")

    dist_flat = dcoeffs.ravel()

    calib_data = {
        "fx": K[0, 0], "fy": K[1, 1],
        "cx": K[0, 2], "cy": K[1, 2],
        "k1": dist_flat[0], "k2": dist_flat[1],
        "p1": dist_flat[2], "p2": dist_flat[3],
        "k3": dist_flat[4] if len(dist_flat) > 4 else 0.0,
        "rms_error": rms,
        "image_width": imagesize[0],
        "image_height": imagesize[1]
    }

    print('Camera Matrix (K):\n', K)
    print('\nLens distortion coefficients:\n', dist_flat)

    with open(options.output, 'w') as f:
        json.dump(calib_data, f, indent=4)
    print(f"\nCalibration data saved to {options.output}")

    print("\nShowing live calibration result. Press 'ESC' or 'q' to exit.")
    h, w = imagesize[1], imagesize[0]

    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, dcoeffs, (w, h), 1, (w, h))
    x, y, roi_w, roi_h = roi

    preview_win = 'Calibration Preview - Raw vs Undistorted'
    cv2.namedWindow(preview_win)

    while True:
        ret, frame = cap.read()
        if not ret: break

        dst = cv2.undistort(frame, K, dcoeffs, None, newcameramtx)
        dst_cropped = dst[y:y+roi_h, x:x+roi_w]

        if dst_cropped.size > 0:
            dst_resized = cv2.resize(dst_cropped, (w, h))
        else:
            dst_resized = dst

        canvas = np.hstack((frame, dst_resized))
        cv2.putText(canvas, "ORIGINAL (Distorted)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(canvas, "CORRECTED (Undistorted)", (w + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow(preview_win, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in [27, ord('q')]:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
