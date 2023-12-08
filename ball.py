import cv2
import numpy as np

# Define the colors of the balls to be tracked (in BGR format)
ball_colors = {'red': (0, 0, 255), 'green': (0, 255, 0), 'blue': (255, 0, 0)}

# Define the number of quadrants and initialize counters
num_quadrants = 4
enter_count = {color: [0] * num_quadrants for color in ball_colors}
exit_count = {color: [0] * num_quadrants for color in ball_colors}

# Read the video file
video_path = "C:/Users/91939/OneDrive/Desktop/AI Assignment video.mp4"
cap = cv2.VideoCapture(video_path)

# Define the video writer for the output
output_path = 'output_video.avi'
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to HSV for color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Iterate through each ball color
    for color, hsv_range in ball_colors.items():
        lower_color = np.array(hsv_range) - np.array([10, 50, 50])
        upper_color = np.array(hsv_range) + np.array([10, 50, 50])

        # Create a mask for the color
        mask = cv2.inRange(hsv, lower_color, upper_color)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # adjust the area threshold based on your scenario
                x, y, w, h = cv2.boundingRect(contour)
                center = (x + w // 2, y + h // 2)

                # Draw a bounding box around the detected ball
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

                # Determine the quadrant based on the ball's center
                quadrant = int(center[0] / (frame.shape[1] / num_quadrants))

                # Check if the ball has entered or exited a quadrant
                if x < 0 or x + w >= frame.shape[1]:
                    exit_count[color][quadrant] += 1
                else:
                    enter_count[color][quadrant] += 1

    # Display the frame
    out.write(frame)
    cv2.imshow('Frame', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

# Display the counts of entering and exiting events for each color and quadrant
for color, enter_values in enter_count.items():
    for quadrant, count in enumerate(enter_values):
        print(f"{color} in Quadrant {quadrant + 1}: Entered {count} times, Exited {exit_count[color][quadrant]} times.")
