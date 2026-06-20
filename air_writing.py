import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# Start webcam
cap = cv2.VideoCapture(0)

canvas = None
prev_x, prev_y = 0, 0

draw_color = (0, 255, 0)
thickness = 5

while True:
    success, frame = cap.read()

    if not success:
        break

    # Flip image
    frame = cv2.flip(frame, 1)

    h, w, c = frame.shape

    # Create blank canvas
    if canvas is None:
        canvas = np.zeros_like(frame)

    # Convert to RGB
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hands
    result = hands.process(rgb)

    if result.multi_hand_landmarks:

        for hand_landmarks in result.multi_hand_landmarks:

            # Index finger tip
            index_finger = hand_landmarks.landmark[8]

            cx = int(index_finger.x * w)
            cy = int(index_finger.y * h)

            # Middle finger tip
            middle_finger = hand_landmarks.landmark[12]

            mx = int(middle_finger.x * w)
            my = int(middle_finger.y * h)

            # Distance between fingers
            distance = np.hypot(mx - cx, my - cy)

            # Draw only when fingers apart
            if distance > 40:

                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = cx, cy

                cv2.line(
                    canvas,
                    (prev_x, prev_y),
                    (cx, cy),
                    draw_color,
                    thickness
                )

                prev_x, prev_y = cx, cy

            else:
                prev_x, prev_y = 0, 0

            # Draw landmarks
            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

    else:
        prev_x, prev_y = 0, 0

    # Merge drawing with webcam
    combined = cv2.addWeighted(frame, 0.7, canvas, 1, 0)

    # Text
    cv2.putText(
        combined,
        "C = Clear | Q = Quit",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.imshow("Air Writing", combined)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    elif key == ord('c'):
        canvas = np.zeros_like(frame)

cap.release()
cv2.destroyAllWindows()