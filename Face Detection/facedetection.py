import cv2
import numpy as np
import pandas as pd

# Set up the camera (use 0 for the built-in camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open the camera.")
    exit()

# Set up the detection parameters
min_area = 1000
max_area = 10000

# Initialize the background subtractor
bg_subtractor = cv2.createBackgroundSubtractorMOG2()

# Counter for detections
detection_count = 0

# Main loop
try:
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break
        
        # Convert the frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply background subtraction
        fg_mask = bg_subtractor.apply(gray)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if min_area < area < max_area:
                # Draw a bounding rectangle around the detected object
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Increment the detection counter
                detection_count += 1
        
        # Display the output
        cv2.imshow('Frame', frame)
        cv2.imshow('Foreground Mask', fg_mask)
        
        # Exit on key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    # Output the total detection count
    print(f"Total detections: {detection_count}")
