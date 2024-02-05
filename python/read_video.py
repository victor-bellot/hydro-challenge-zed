import sys
import cv2


def display_bgr_video(video_path):
    # Open the video file
    cap = cv2.VideoCapture(video_path)

    # Check if the video file is opened successfully
    if not cap.isOpened():
        print("Error: Could not open the video file.")
        return

    # Get video FPS
    fps = cap.get(5)

    while True:
        # Read a frame from the video file
        ret, frame = cap.read()

        # Break the loop if no more frames are available
        if not ret:
            break

        # Display the frame
        cv2.imshow('BGR Video', frame)
        cv2.waitKey(int(1e3 / fps))

    # Release the VideoCapture object
    cap.release()

    # Close all OpenCV windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    video_name = 'recording'
    if len(sys.argv) > 1:
        video_name = sys.argv[1]

    # Call the function to display the BGR video
    display_bgr_video('../build/' + video_name + '.avi')
