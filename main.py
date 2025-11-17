import cv2
import numpy as np


def motion_to_color(magnitude):
    """
    Convert motion magnitude to color:
    Small motion   -> green
    Medium motion  -> orange
    Large motion   -> red
    """
    if magnitude < 2:
        return (0, 255, 0)        # green
    elif magnitude < 5:
        return (0, 165, 255)      # orange
    else:
        return (0, 0, 255)        # red


def main():

    # ----------------------------------------
    # INPUT SELECTION
    # ----------------------------------------
    print("Select input source:")
    print("1. Webcam")
    print("2. Video file")
    choice = input("Enter 1 or 2: ")

    if choice == "2":
        path = input("Enter video file path: ")
        cap = cv2.VideoCapture(path)
    else:
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Unable to open video source.")
        return

    # ----------------------------------------
    # Feature detection parameters
    # ----------------------------------------
    feature_params = dict(
        maxCorners=600,          # DENSE
        qualityLevel=0.01,
        minDistance=4,
        blockSize=7
    )

    # ----------------------------------------
    # LK parameters
    # ----------------------------------------
    lk_params = dict(
        winSize=(21, 21),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    )

    # ----------------------------------------
    # Read first frame
    # ----------------------------------------
    ret, old_frame = cap.read()
    if not ret:
        print("Could not read first frame.")
        return

    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)

    # Detect initial features
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    # Mask for trails
    mask = np.zeros_like(old_frame)

    # ----------------------------------------
    # VIDEO WRITER
    # ----------------------------------------
    height, width = old_frame.shape[:2]
    fps = cap.get(cv2.CAP_PROP_FPS)

    # fallback FPS for webcam
    if fps == 0:
        fps = 20

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("klt_output.mp4", fourcc, fps, (width, height))

    print("Saving output to: klt_output.mp4")

    # ----------------------------------------
    # MAIN LOOP
    # ----------------------------------------
    while True:

        ret, frame = cap.read()
        if not ret:
            break

        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Optical Flow
        p1, status, err = cv2.calcOpticalFlowPyrLK(
            old_gray, frame_gray, p0, None, **lk_params
        )

        # Re-detect features if tracking fails
        if p1 is None or status is None:
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
            mask = np.zeros_like(frame)
            old_gray = frame_gray.copy()
            continue

        good_new = p1[status == 1]
        good_old = p0[status == 1]

        # If no points available
        if len(good_new) == 0:
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)
            old_gray = frame_gray.copy()
            continue

        # --------------------------------
        # FADING TRAIL EFFECT
        # --------------------------------
        mask = (mask * 0.85).astype(np.uint8)

        # --------------------------------
        # DRAW ARROWS + TRAILS
        # --------------------------------
        for new_pt, old_pt in zip(good_new, good_old):

            x_new, y_new = new_pt.ravel()
            x_old, y_old = old_pt.ravel()

            dx = x_new - x_old
            dy = y_new - y_old
            mag = np.sqrt(dx * dx + dy * dy)

            color = motion_to_color(mag)
            # -------- CONSTANT ARROW SIZE --------
            arrow_length = 40     # FIXED arrow size (you can change)
            if mag > 0:
                ux = dx / mag     # unit vector
                uy = dy / mag
            else:
                ux = uy = 0

            # End point with constant length
            end_x = int(x_old + ux * arrow_length)
            end_y = int(y_old + uy * arrow_length)
            
            
            if mag < 2:
                # GREEN → draw a fixed-size square instead of an arrow
                square_size = 2
                frame = cv2.rectangle(
                frame,
                (int(x_old - square_size), int(y_old - square_size)),
                (int(x_old + square_size), int(y_old + square_size)),
                (0, 255, 0),
                2
            )
            else:

             # ORANGE + RED → draw arrows (constant size)
                frame = cv2.arrowedLine(
                    frame,
                    (int(x_old), int(y_old)),
                    (end_x, end_y),
                    color,
                    2,            
                    tipLength=0.1  # Larger arrow head
            )

        # --------------------------------
        # MERGE LAYERS
        # --------------------------------
        output = cv2.addWeighted(frame, 0.6, mask, 1.0, 0)

        # Display
        cv2.imshow("KLT Optical Flow (Arrows + Trails + Color)", output)

        # Save output frame
        out.write(output)

        # Key controls
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == ord('r'):  # reset features
            mask = np.zeros_like(frame)
            p0 = cv2.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)

        # Update
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    print("Done. Saved file: klt_output.mp4")


if __name__ == "__main__":
    main()
