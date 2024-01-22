import time
from tools import *


def main(video_name, start_time=0, is_stereo=True):
    if is_stereo:
        n_column = 2
        video_path = 'stereo_video/' + video_name
    else:
        n_column = 1
        video_path = 'video/' + video_name

    # Create a VideoCapture object
    cap = cv2.VideoCapture(video_path)

    # Check if the video file was successfully opened
    if not cap.isOpened():
        print("Error: Could not open video file.")
        exit()

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / n_column)

    # Define colors in BGR
    color_of = (255, 0, 0)  # BLUE
    color_grad = (0, 0, 255)  # RED
    color_horizon = (0, 255, 0)  # GREEN
    color_box = (255, 0, 255)  # PURPLE

    horizontal_ceil = 16  # what's considered as horizontal

    # Used to draw the horizontal line : (2, 1, 1)
    border_x = np.array([[[0.]], [[width - 1.]]])
    border_y = np.full((2, 1, 1), height / 2)

    # Blur each frame before processing them
    window_gaussian = (7, 7)

    # Dilatation is applied on active pixel -> used for computing the saliency mask
    dilatation_kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(3, 3))

    # Parameters for corner detection
    features_params = {
        'maxCorners': 10000,
        'qualityLevel': 0.01,
        'minDistance': 0.1
    }

    # Parameters for pyramidal Lucas-Kanade optical flow
    lk_params = {
        'winSize': (15, 15),
        'maxLevel': 2,
        'criteria': (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    }

    # Tracking window
    wx = width / 2.
    wy = height / 2.
    ws = min(wx, wy)

    # Set duration (in seconds)
    duration = 40
    dt = 1. / fps

    # Calculate the corresponding frame numbers
    n_frame = duration * fps
    start_frame = int(start_time * fps)

    # Set the initial frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Define variables to initialize later
    old_frame = None
    points_old_frame = []
    horizontal_mask = np.ones(shape=(height - int(height / 3), width), dtype=np.uint8)

    # Read and display frames within the specified time range
    for _ in range(n_frame):
        t_loop = time.time()

        # Read a frame from the video
        ret, frame_bgr = cap.read()

        # Check if the video has ended
        if not ret:
            break

        # Only consider the left camera flow and the lower part
        frame_bgr = frame_bgr[int(height / 3):, :width]
        frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        # Apply a blur filter
        frame = cv2.GaussianBlur(frame, window_gaussian, 0)

        # Where to look
        saliency_mask = compute_saliency_mask(frame_bgr)
        saliency_mask = cv2.dilate(saliency_mask, dilatation_kernel)

        # Estimate a line for the horizon
        hpx, hpy, ransac_border_y = estimate_horizon(frame)

        # Filter out non horizontal lines
        y_0, y_w = ransac_border_y.flatten()
        if abs(y_w - y_0) < horizontal_ceil:
            border_y = ransac_border_y.reshape(-1, 1, 1)
            horizontal_mask = np.zeros_like(frame, dtype=np.uint8)
            horizontal_mask[int(max(y_0, y_w)):] = 1

        # Draw points and line on a separate image
        canvas = frame_bgr.copy()

        if (old_frame is not None) and (points_old_frame is not None):
            points_new_frame, status, err = cv2.calcOpticalFlowPyrLK(old_frame, frame,
                                                                     points_old_frame,
                                                                     None, **lk_params)

            for p0, p1 in zip(points_old_frame[status == 1], points_new_frame[status == 1]):
                x0, y0 = p0.ravel()
                x1, y1 = p1.ravel()
                cv2.line(canvas, (int(x1), int(y1)), (int(x0), int(y0)), color_of, 3)

        # Current frame becomes the old one
        old_frame = frame.copy()

        # Compute good feature point to keep track of
        mask = saliency_mask & horizontal_mask
        points_old_frame = cv2.goodFeaturesToTrack(frame, mask=mask, **features_params)

        # Update tracking window
        wx, wy, ws = update_tracking_window(wx, wy, ws, points_old_frame)
        cv2.rectangle(canvas, (int(wx-ws), int(wy-ws)), (int(wx+ws), int(wy+ws)), color_box, thickness=2)

        # Scatter gradient points on the image
        for (x, y) in zip(hpx, hpy):
            cv2.circle(canvas, (int(x), int(y)), 3, color_grad, -1)

        # Draw the horizontal line on the image
        ransac_points = np.concatenate((border_x, border_y), axis=2).astype(np.int32)
        cv2.polylines(canvas, [ransac_points], isClosed=False, color=color_horizon, thickness=2)

        debug = cv2.cvtColor(mask * 255, cv2.COLOR_GRAY2BGR)

        # Stack images vertically or horizontally
        if height < width:
            layout = np.vstack((canvas, debug))
        else:
            layout = np.hstack((canvas, debug))

        # Show layout
        cv2.imshow('Canvas & Debug', layout)

        # Pause the loop if SPACE is pressed
        paused = False
        if (cv2.waitKey(25) & 0xFF) == ord(' '):
            paused = True
            while (cv2.waitKey(25) & 0xFF) != ord(' '):
                continue

        t_remaining = EPSILON if paused else dt - (time.time() - t_loop)
        if t_remaining > 0:
            plt.pause(t_remaining)
        else:
            print("Out of time of", -t_remaining, 'seconds.')
            plt.pause(EPSILON)

    # Release the VideoCapture object and close the window
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # main('duck_close.avi')
    main('goodbye.mp4', start_time=10, is_stereo=False)
