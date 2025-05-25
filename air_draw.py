import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import cv2
import mediapipe as mp
import numpy as np

# Initialize camera and mediapipe
cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

# Drawing setup
brush_color = (255, 0, 255)
tool = "Brush"
canvas = None
xp, yp = 0, 0

# Color swatches
color_swatches = [
    (10, 50, 60, 100, (255, 0, 255)),  # Magenta
    (70, 50, 120, 100, (255, 0, 0)),   # Blue
    (130, 50, 180, 100, (0, 255, 0)),  # Green
    (190, 50, 240, 100, (0, 0, 255)),  # Red
]

# Count fingers
def count_fingers(hand_landmarks):
    tip_ids = [4, 8, 12, 16, 20]
    fingers = []
    if hand_landmarks.landmark[tip_ids[0]].x < hand_landmarks.landmark[tip_ids[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)
    for i in range(1, 5):
        if hand_landmarks.landmark[tip_ids[i]].y < hand_landmarks.landmark[tip_ids[i] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

# Gesture helpers
def is_thumb_up(fingers):
    return fingers[0] == 1 and sum(fingers[1:]) == 0

def is_thumb_down_with_index(fingers):
    return fingers[0] == 0 and fingers[1] == 1 and sum(fingers[2:]) == 0

def is_palm_open(fingers):
    return sum(fingers) == 5

while True:
    success, img = cap.read()
    if not success:
        break
    img = cv2.flip(img, 1)
    if canvas is None:
        canvas = np.zeros_like(img)

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            lmList = []
            for id, lm in enumerate(handLms.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((cx, cy))

            if lmList:
                fingers = count_fingers(handLms)

                # Gestures to switch tools
                if is_thumb_up(fingers):
                    tool = "Eraser"
                elif is_thumb_down_with_index(fingers):
                    tool = "Brush"
                elif is_palm_open(fingers):
                    canvas = np.zeros_like(img)  # Clear all

                # Check color swatch clicks
                ix, iy = lmList[8]
                for (x1, y1, x2, y2, color) in color_swatches:
                    if x1 < ix < x2 and y1 < iy < y2:
                        brush_color = color
                        tool = "Brush"

                # Drawing or erasing
                if sum(fingers[1:]) == 1:
                    cx, cy = lmList[8]
                    if xp == 0 and yp == 0:
                        xp, yp = cx, cy
                    if tool == "Brush":
                        cv2.line(canvas, (xp, yp), (cx, cy), brush_color, 15)
                    elif tool == "Eraser":
                        cv2.line(canvas, (xp, yp), (cx, cy), (0, 0, 0), 50)
                    xp, yp = cx, cy
                else:
                    xp, yp = 0, 0

            mp_draw.draw_landmarks(img, handLms, mp_hands.HAND_CONNECTIONS)

    # Merge canvas and webcam frame
    img_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, img_inv = cv2.threshold(img_gray, 20, 255, cv2.THRESH_BINARY_INV)
    img_inv = cv2.cvtColor(img_inv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, img_inv)
    img = cv2.bitwise_or(img, canvas)

    # Draw UI
    for (x1, y1, x2, y2, color) in color_swatches:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 255), 2)

    cv2.putText(img, f"Tool: {tool}", (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
    cv2.rectangle(img, (10, 150), (60, 200), brush_color if tool == "Brush" else (0, 0, 0), -1)

    cv2.putText(img, "Thumb Up: Eraser | Thumb Down+Index: Brush | Palm: Clear", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 50, 50), 2)

    # Show
    cv2.imshow("AirDraw - Brush, Eraser, Colors, Clear", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
