# Import necessary libraries
import cv2  # OpenCV for handling video capture and image processing
import os  # OS module for file and directory operations

# -------------------
# Settings for capturing images
# -------------------

# Define paths for saving captured images
output_left_folder = "captured_images/left"  # Folder for left camera images
output_right_folder = "captured_images/right"  # Folder for right camera images

# Create directories if they do not exist
os.makedirs(output_left_folder, exist_ok=True)  # Ensure left folder exists
os.makedirs(output_right_folder, exist_ok=True)  # Ensure right folder exists

# Connect to the stereo camera
stereo_camera_index = 0  # Index for accessing the stereo camera
cap = cv2.VideoCapture(stereo_camera_index)  # Open the stereo camera

# Check if the camera is successfully opened
if not cap.isOpened():
    print("Unable to open the stereo camera.")  # Error message if camera fails
    exit()

# Counter to track the number of images captured
image_counter = 1
max_images = 30  # Set the maximum number of images to capture for each side

# Display instructions to the user
print("Press 'c' to capture an image or 'q' to quit.")

# Infinite loop for capturing images
while True:
    # Read a frame from the stereo camera
    ret, frame = cap.read()  # `ret` indicates success, `frame` is the image data

    # Check if the frame was successfully read
    if not ret:
        print("Unable to read frames from the stereo camera.")  # Error message
        break

    # Split the stereo frame into left and right images
    height, width, _ = frame.shape  # Get dimensions of the frame
    mid = width // 2  # Calculate the middle point of the frame
    frame_left = frame[:, :mid]  # Extract the left side of the frame
    frame_right = frame[:, mid:]  # Extract the right side of the frame

    # Concatenate the left and right images for display purposes
    stereo_concat = cv2.hconcat([frame_left, frame_right])  # Horizontally stack images

    # Display the number of captured images on the frame
    text = f"{image_counter}/{max_images} (Press 'c' to capture, 'q' to quit)"
    font = cv2.FONT_HERSHEY_SIMPLEX  # Font style
    cv2.putText(stereo_concat, text, (10, 30), font, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the concatenated stereo image in a window
    cv2.imshow("Stereo Camera Capture", stereo_concat)

    # Wait for user input (1 ms delay)
    key = cv2.waitKey(1) & 0xFF  # Capture the key press

    if key == ord('c'):  # If 'c' is pressed, capture an image
        if image_counter <= max_images:
            # Define file paths for left and right images
            left_image_path = os.path.join(output_left_folder, f"left_{image_counter:03d}.png")
            right_image_path = os.path.join(output_right_folder, f"right_{image_counter:03d}.png")

            # Save the left and right images to their respective folders
            cv2.imwrite(left_image_path, frame_left)
            cv2.imwrite(right_image_path, frame_right)

            # Print confirmation message
            print(f"Captured: {left_image_path} and {right_image_path}")
            image_counter += 1  # Increment the image counter

            # Stop capturing if the maximum number of images is reached
            if image_counter > max_images:
                print("Captured maximum images. Exiting capture process.")
                break

    elif key == ord('q'):  # If 'q' is pressed, exit the capture process
        print("Exiting capture process.")
        break

# Release the camera resource and close OpenCV windows
cap.release()  # Release the camera for other applications
cv2.destroyAllWindows()  # Close all OpenCV windows