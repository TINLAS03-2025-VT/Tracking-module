import json
import cv2
import numpy as np
import apriltag
import defines


def main():
    result = {
        "event": "self_test_started",
        "opencv_version": cv2.__version__,
        "apriltag_imported": True,
        "defines_imported": True,
        "available_camera_profiles": [
            name for name in dir(defines)
            if not name.startswith("_")
        ],
    }

    print(json.dumps(result), flush=True)

    detector = apriltag.apriltag(
        family="tagStandard41h12",
        threads=1,
        maxhamming=1,
        decimate=2.0,
        blur=0.0,
        refine_edges=True,
        debug=False,
    )

    dummy_image = np.zeros((480, 640), dtype=np.uint8)
    detections = detector.detect(dummy_image)

    print(json.dumps({
        "event": "detector_test_complete",
        "dummy_image_shape": list(dummy_image.shape),
        "detections_found": len(detections),
        "status": "ok"
    }), flush=True)


if __name__ == "__main__":
    main()
