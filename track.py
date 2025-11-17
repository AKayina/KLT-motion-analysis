import cv2
import numpy as np

def main():
    # INPUT CHOICE
    print("Select input source:")
    print("1. Webcam")
    print("2. Video File")
    choice = input("Enter 1 or 2: ")

    if choice == "2":
        video_path = input("Enter video file path: ")
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(0)

    # Check if opened
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    # Prepare Video Writer for saving output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = None  # initialize later after reading first frame

    # Corner detector (Shi-Tomasi)
    feature_params = dict(
        maxCorners=100,
        qualityLevel=0.3,
        minDistance=7,
        blockSize=7
    )

    # ORB for descriptor extraction
    orb = cv2.ORB_create()

    # BF-Matcher
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        print("Error: Could not read first frame.")
        return

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)

    # Initialize VideoWriter after getting frame size
    height, width = prev_frame.shape[:2]
    out = cv2.VideoWriter("output_feature_tracking.mp4", fourcc, 20.0, (width, height))

    # Detect corners in first frame
    p0 = cv2.goodFeaturesToTrack(prev_gray, **feature_params)
    p0 = p0.astype(int)

    # Convert corners into keypoints for ORB
    kp_prev = [cv2.KeyPoint(float(x), float(y), 7) for [[x, y]] in p0]
    kp_prev, des_prev = orb.compute(prev_gray, kp_prev)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect new corners
        p1 = cv2.goodFeaturesToTrack(gray, **feature_params)
        if p1 is None:
            out.write(frame)
            cv2.imshow("Feature Tracking", frame)
            if cv2.waitKey(1) == 27:
                break
            continue

        p1 = p1.astype(int)
        kp_new = [cv2.KeyPoint(float(x), float(y), 7) for [[x, y]] in p1]
        kp_new, des_new = orb.compute(gray, kp_new)

        # Match descriptors
        if des_new is not None and des_prev is not None:
            matches = bf.match(des_prev, des_new)
            matches = sorted(matches, key=lambda x: x.distance)

            # Draw ONLY matched features as squares
            for m in matches:
                x, y = map(int, kp_new[m.trainIdx].pt)

                size = 6
                top_left = (x - size, y - size)
                bottom_right = (x + size, y + size)

                cv2.rectangle(frame, top_left, bottom_right, (0, 0, 255), 2)

        # Show the frame
        cv2.imshow("Feature Tracking", frame)

        # Save the frame to output file
        out.write(frame)

        # Update for next frame
        prev_gray = gray.copy()
        kp_prev, des_prev = kp_new, des_new

        if cv2.waitKey(1) == 27:
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Output saved as output_feature_tracking.mp4")


if __name__ == "__main__":
    main()
