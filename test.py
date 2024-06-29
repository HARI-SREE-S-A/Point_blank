import cv2

# Load the cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')




# Function to detect the approximate midpoint between eyes
def detect_midpoint(face_region, face_coords):
    (x, y, w, h) = face_coords

    # Define the region where eyes are likely to be
    eye_region_y_start = int(h * 0.3)
    eye_region_y_end = int(h * 0.5)
    eye_region_x_start = int(w * 0.15)
    eye_region_x_end = int(w * 0.85)

    # Calculate the midpoint between the eye regions
    midpoint_x = (eye_region_x_start + eye_region_x_end) // 2
    midpoint_y = (eye_region_y_start + eye_region_y_end) // 2

    # Adjust for overall face coordinates
    midpoint = (x + midpoint_x, y + midpoint_y)
    return midpoint


# Start video capture
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=10,
        minSize=(50, 50),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    target_locked = False

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Extract the face region
        face_region = frame[y:y + h, x:x + w]

        # Find the approximate midpoint between the eyes
        midpoint = detect_midpoint(face_region, (x, y, w, h))

        if midpoint:
            # Draw the smaller target at the midpoint
            cv2.circle(frame, midpoint, 5, (0, 255, 0), 1)
            cv2.circle(frame, midpoint, 10, (0, 255, 0), 1)
            cv2.line(frame, (midpoint[0] - 10, midpoint[1]), (midpoint[0] + 10, midpoint[1]), (0, 255, 0), 1)
            cv2.line(frame, (midpoint[0], midpoint[1] - 10), (midpoint[0], midpoint[1] + 10), (0, 255, 0), 1)

            # Set the flag for target locked
            target_locked = True

    # Display the "Target is locked" message if the target is detected
    if target_locked:
        cv2.putText(frame, "Target is locked", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Face Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close windows
cap.release()
cv2.destroyAllWindows()

