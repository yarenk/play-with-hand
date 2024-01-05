import cv2
import numpy as np
import mediapipe as mp
import pyautogui
from time import sleep

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def find_distance(a, b, frame):
    x1, y1 = int(a.x * frame.shape[1]), int(a.y * frame.shape[0])
    x2, y2 = int(b.x * frame.shape[1]), int(b.y * frame.shape[0])

    cv2.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance

cap = cv2.VideoCapture(0)

with mp_hands.Hands(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame.flags.writeable = False
        results = hands.process(frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 0, 255)),
                    mp_drawing.DrawingSpec(color=(0, 255, 0))
                )

                index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]

                cv2.circle(frame, (int(index_tip.x), int(index_tip.y)), 5, (0, 255, 0), cv2.FILLED)
                cv2.circle(frame, (int(thumb_tip.x), int(thumb_tip.y)), 5, (0, 255, 0), cv2.FILLED)

                distance = find_distance(thumb_tip, index_tip, frame)
                if distance <= 60 :
                    pyautogui.press('Space')
                elif distance >= 120 :
                    pyautogui.press('S')
                cv2.putText(frame, f'Distance: {int(distance)}', (50, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2)

        frame.flags.writeable = True
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow('Hand Tracking', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


cap.release()
cv2.destroyAllWindows()