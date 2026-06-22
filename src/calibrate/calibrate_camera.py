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

    options = parser.parse_args()

    patternsize = (options.cols, options.rows)
    sz = options.size

    x = np.arange(patternsize[0])*sz
    y = np.arange(patternsize[1])*sz

    xgrid, ygrid = np.meshgrid(x, y)
    zgrid = np.zeros_like(xgrid)
    opoints_single = np.dstack((xgrid, ygrid, zgrid)).reshape((-1, 1, 3)).astype(np.float32)

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
    print("Capturing will run for 30 seconds. Press 'q' to stop early.")

    duration = 30
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
                ipoints.append(corners)
                opoints.append(opoints_single)
                last_sample_time = current_time
                print("Saved screenshot {} | Remaining: {}s".format(len(ipoints), remaining_time))

        cv2.putText(display, "Time Left: {}s".format(remaining_time), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        cv2.putText(display, "Snapshots: {}".format(len(ipoints)), (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        cv2.imshow(win, display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("\nStopped early by user. Starting calibration...")
            break

    cap.release()
    cv2.destroyAllWindows()

    if len(ipoints) < 3:
        print("Error: Not enough chessboard patterns detected to perform calibration (Minimum 3 needed).")
        return

    print("\nCalibrating camera matrix based on {} frames...".format(len(ipoints)))

    flags = (cv2.CALIB_ZERO_TANGENT_DIST |
             cv2.CALIB_FIX_K1 |
             cv2.CALIB_FIX_K2 |
             cv2.CALIB_FIX_K3 |
             cv2.CALIB_FIX_K4 |
             cv2.CALIB_FIX_K5 |
             cv2.CALIB_FIX_K6)

    retval, K, dcoeffs, rvecs, tvecs = cv2.calibrateCamera(
        opoints, ipoints, imagesize,
        cameraMatrix=None,
        distCoeffs=np.zeros(5),
        flags=flags
    )

    assert(np.all(dcoeffs == 0))

    fx = K[0,0]
    fy = K[1,1]
    cx = K[0,2]
    cy = K[1,2]

    params = (fx, fy, cx, cy)

    print()
    print('all units below measured in pixels:')
    print('  fx = {}'.format(fx))
    print('  fy = {}'.format(fy))
    print('  cx = {}'.format(cx))
    print('  cy = {}'.format(cy))
    print()
    print('pastable into Python:')
    print('  fx, fy, cx, cy = {}'.format(repr(params)))
    print()

if __name__ == '__main__':
    main()
