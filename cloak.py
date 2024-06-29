import numpy as np
import cv2
import time

# Initialize the webcam
cap = cv2.VideoCapture(0)
time.sleep(2)

# Capture the background
background = 0
for i in range(30):
    ret, background = cap.read()

# Define the kernel for morphological operations
kernel = np.ones((3, 3), np.uint8)

while cap.isOpened():
    ret, img = cap.read()

    if not ret:
        break

    # Convert the frame from BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define the range for red color in HSV
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])

    # Create masks to detect red color
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

    mask1 = mask1 + mask2

    # Perform morphological operations to remove noise
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, kernel, iterations=2)
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, kernel, iterations=1)

    # Create an inverse mask to segment out the red color
    mask2 = cv2.bitwise_not(mask1)

    # Segment the red color part out of the captured frame using bitwise AND
    res1 = cv2.bitwise_and(background, background, mask=mask1)

    # Create the final output by combining the background with the current frame where the red color is not detected
    res2 = cv2.bitwise_and(img, img, mask=mask2)
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

    # Display the final output
    cv2.imshow('Eureka!!', final_output)

    # Exit on pressing 'ESC'
    k = cv2.waitKey(10)
    if k == 27:
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
