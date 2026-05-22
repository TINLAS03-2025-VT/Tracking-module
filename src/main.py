import apriltag
import cv2
import numpy as np
import defines as defines

# Initialize detector
detector = apriltag.apriltag(
	family='tagStandard41h12',  # Tag family
	threads=10,                 # Number of threads
	maxhamming=1,               # Maximum hamming distance for error correction
	decimate=2.0,               # Image downsampling factor
	blur=0.0,                   # Gaussian blur sigma
	refine_edges=True,          # Refine quad edges
	debug=False                 # Debug mode
)

# Camera calibration parameters
cam_cal = defines.microsoft_cam

cap = cv2.VideoCapture(5)

SCALE_X = 200
SCALE_Y = 200

DEST_TAG_0 = np.array([0, 0], dtype=np.float32)
DEST_TAG_1 = np.array([SCALE_X, SCALE_Y], dtype=np.float32)

while True:
	ret, frame = cap.read()
	if not ret:
		break

	# Convert to grayscale
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# Detect tags
	detections = detector.detect(gray)

	# Dictionary to store the centers of tags found in this specific frame
	frame_tags = {}

	# Draw results
	for det in detections:
		print("Detected")

		tag_id = int(det['id'])
		center = tuple((det['center']).astype(int))
		frame_tags[tag_id] = center

		# Draw corners
		corners = det['lb-rb-rt-lt'].astype(int)
		cv2.polylines(frame, [corners], True, (0, 255, 0), 2)

		# Draw center
		cv2.circle(frame, center, 5, (0, 0, 255), -1)

		# Draw ID
		cv2.putText(frame, str(tag_id), center,
					cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

	print(f"frame_tags: {frame_tags}")

	if 0 in frame_tags and 1 in frame_tags:
		p0 = np.array(frame_tags[0], dtype=np.float32)
		p1 = np.array(frame_tags[1], dtype=np.float32)

		v = p1 - p0
		p2 = p0 + np.array([-v[1], v[0]])

		src_pts = np.array([p0, p1, p2], dtype=np.float32)

		v_dest = DEST_TAG_1 - DEST_TAG_0
		d2 = DEST_TAG_0 + np.array([-v_dest[1], v_dest[0]])

		dest_pts = np.array([DEST_TAG_0, DEST_TAG_1, d2], dtype=np.float32)

		transformation_matrix = cv2.getAffineTransform(src_pts, dest_pts)
		rev_matrix = cv2.invertAffineTransform(transformation_matrix)

		def grid_to_pixel(gx, gy):
			homog = np.array([gx, gy, 1.0])
			pix = np.dot(rev_matrix, homog)
			return (int(pix[0]), int(pix[1]))

		c_0_0   = grid_to_pixel(0, 0)
		c_200_0 = grid_to_pixel(SCALE_X, 0)
		c_200_200 = grid_to_pixel(SCALE_X, SCALE_Y)
		c_0_200 = grid_to_pixel(0, SCALE_Y)

		cv2.line(frame, c_0_0, c_200_0, (255, 255, 0), 2)
		cv2.line(frame, c_200_0, c_200_200, (255, 255, 0), 2)
		cv2.line(frame, c_200_200, c_0_200, (255, 255, 0), 2)
		cv2.line(frame, c_0_200, c_0_0, (255, 255, 0), 2)

		for i in range(50, SCALE_X, 50):
			# Vertical lines
			cv2.line(frame, grid_to_pixel(i, 0), grid_to_pixel(i, SCALE_Y), (180, 180, 180), 1)
			# Horizontal lines
			cv2.line(frame, grid_to_pixel(0, i), grid_to_pixel(SCALE_X, i), (180, 180, 180), 1)

		# Label the origins on the grid overlay
		cv2.putText(frame, "(0,0)", c_0_0, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
		cv2.putText(frame, f"({SCALE_X},{SCALE_Y})", c_200_200, cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

		# Estimate and print pose
		pose = detector.estimate_tag_pose(det, defines.tagsize, cam_cal["fx"], cam_cal["fy"], cam_cal["cx"], cam_cal["cy"])
		print(f"Tag {det['id']} location: {pose}")

		for target_id in [2, 3]:
			if target_id in frame_tags:
				pixel_coords = frame_tags[target_id]

				# Reshape to format required by OpenCV coordinate transformation matrix multiplication
				homogenous_coord = np.array([pixel_coords[0], pixel_coords[1], 1.0])
				mapped_space = np.dot(transformation_matrix, homogenous_coord)

				mapped_x = mapped_space[0]
				mapped_y = mapped_space[1]

				print(f"Tag {target_id} -> Mapped 2D Pos: X={mapped_x:.2f}, Y={mapped_y:.2f}")

				# Display the mapped coordinates on the video stream for validation
				cv2.putText(frame, f"Mapped: ({int(mapped_x)}, {int(mapped_y)})",
							(int(pixel_coords[0]), int(pixel_coords[1]) + 20),
							cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
	else:
		# Give a warning overlay if your reference baseline tags are missing
		cv2.putText(frame, "Baseline tags (0 or 1) missing!", (10, 30),
					cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

	cv2.imshow('AprilTag Detection', frame)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

cap.release()
cv2.destroyAllWindows()
