import cv2
import numpy as np

# Video source (use your video file path or 0 for webcam)
video_path = 'assets/video.mp4'
cap = cv2.VideoCapture(video_path)

# Minimum dimensions for the rectangle (vehicles)
min_width_rectangle = 80
min_height_rectangle = 80

# Line position for vehicle counting
count_line_position = 550

# Background subtractor for motion detection
background_subtractor = cv2.createBackgroundSubtractorMOG2(history=100, varThreshold=40)

# Helper function to calculate the center of a rectangle
def calculate_center(x, y, w, h):
    cx = x + w // 2
    cy = y + h // 2
    return cx, cy

# Variables to track detected objects
detections = []
offset = 6  # Allowable error in pixels for line crossing
vehicle_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 5)

    # Apply the background subtractor
    mask = background_subtractor.apply(blur)

    # Morphological transformations to remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilated = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the counting line
    cv2.line(frame, (25, count_line_position), (1200, count_line_position), (255, 0, 0), 3)

    # Process each contour
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        if w >= min_width_rectangle and h >= min_height_rectangle:
            # Draw the bounding box and label
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            center = calculate_center(x, y, w, h)
            detections.append(center)
            cv2.circle(frame, center, 4, (0, 0, 255), -1)

            # Check if the center has crossed the counting line
            for (cx, cy) in detections:
                if count_line_position - offset < cy < count_line_position + offset:
                    vehicle_count += 1
                    cv2.line(frame, (25, count_line_position), (1200, count_line_position), (0, 127, 255), 3)
                    detections.remove((cx, cy))
                    print(f"Vehicle No: {vehicle_count}")

    # Display the vehicle count on the frame
    cv2.putText(frame, f"Vehicle Count: {vehicle_count}", (450, 70), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 5)

    # Show the frame
    cv2.imshow('Vehicle Detection', frame)

    # Break the loop on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

