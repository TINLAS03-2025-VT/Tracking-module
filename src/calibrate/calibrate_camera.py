from __future__ import print_function
import numpy as np
import cv2
import argparse
import time

def main():

    parser = argparse.ArgumentParser(
        description='calibrate camera intrinsics using OpenCV via live video stream')

    parser.add_argument('-i', '--camera-index', type=int, default=0,
                        help='Index of the camera to use (default: 0)')

    parser.add_argument('-r', '--rows', metavar='N', type=int,
                        required=True,
                        help='# of chessboard corners in vertical direction')

    parser.add_argument('-c', '--cols', metavar='N', type=int,
                        required=True,
                        help='# of chessboard corners in horizontal direction')

    parser.add_argument('-s', '--size', metavar='NUM', type=float, default=1.0,
                        help='chessboard square size in user-chosen units (should not affect results)')

    parser.add_argument('-t', '--time', metavar='N', type=int, default=30, help='time the calibration runs for')

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
    win = 'Calibrate - Live Stream'
    cv2.namedWindow(win)

    ipoints = []
    opoints = []

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

        if len(rgb.shape) == 3:
            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
        else:
            gray = rgb

        retval, corners = cv2.findChessboardCorners(gray, patternsize, None)
        display = rgb.copy()

        if retval:
            cv2.drawChessboardCorners(display, patternsize, corners, retval)

            current_time = time.time()
            if current_time - last_sample_time > 0.5:
                corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term_criteria)

                ipoints.append(corners2)
                opoints.append(opoints_single)

                last_sample_time = current_time
                print("Saved screenshot {} | Remaining: {}s".format(len(ipoints), remaining_time))

        cv2.putText(display, "Time Left: {}s".format(remaining_time), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(display, "Snapshots: {}".format(len(ipoints)), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow(win, display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nStopped early by user. Starting calibration...")
            break

    cv2.destroyWindow(win)

    if len(ipoints) < 3:
        print("Error: Not enough chessboard patterns detected to perform calibration (Minimum 3 needed).")
        cap.release()
        return

    print("\nCalibrating camera matrix based on {} frames...".format(len(ipoints)))

    retval, K, dcoeffs, rvecs, tvecs = cv2.calibrateCamera(
        opoints, ipoints, imagesize,
        cameraMatrix=None,
        distCoeffs=None,
        flags=0
    )

    dist_flat = dcoeffs.ravel()
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    k1, k2, p1, p2, k3 = dist_flat[:5]

    print('\nAll units below measured in pixels:')
    print('  fx = {}'.format(fx))
    print('  fy = {}'.format(fy))
    print('  cx = {}'.format(cx))
    print('  cy = {}'.format(cy))
    print('\nLens distortion coefficients:')
    print('  k1 = {}\n  k2 = {}\n  p1 = {}\n  p2 = {}\n  k3 = {}'.format(k1, k2, p1, p2, k3))
    print('\nPastable profile dictionary for your tracking application:')
    print("""cam_cal = {{
    "fx": {},
    "fy": {},
    "cx": {},
    "cy": {},
    "k1": {},
    "k2": {},
    "p1": {},
    "p2": {},
    "k3": {}
}}""".format(fx, fy, cx, cy, k1, k2, p1, p2, k3))

    print("\nShowing live calibration result. Press 'ESC' or 'q' on the preview window to exit.")

    h, w = imagesize[1], imagesize[0]
    # Compute the optimal new camera matrix to handle undistorted image boundaries
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(K, dcoeffs, (w, h), 1, (w, h))
    x, y, roi_w, roi_h = roi

    preview_win = 'Calibration Preview - Raw vs Undistorted'
    cv2.namedWindow(preview_win)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Undistort the live frame
        dst = cv2.undistort(frame, K, dcoeffs, None, newcameramtx)

        # Crop the image based on the calculated optimal ROI
        dst_cropped = dst[y:y+roi_h, x:x+roi_w]

        # Resize cropped image back to original size for a clean side-by-side view
        if dst_cropped.size > 0:
            dst_resized = cv2.resize(dst_cropped, (w, h))
        else:
            dst_resized = dst # Fallback if ROI is empty

        # Stack raw and corrected images side-by-side
        canvas = np.hstack((frame, dst_resized))

        # Labels
        cv2.putText(canvas, "ORIGINAL (Distorted)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "CORRECTED (Undistorted)", (w + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow(preview_win, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27: # 'q' or ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
