import cv2
import mediapipe as mp
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Finger tip landmarks
finger_tips = [8, 12, 16, 20]
thumb_tip = 4

# Webcam
cap = cv2.VideoCapture(0)

last_output = None
last_time = time.time()

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    output = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            fingers_up = 0

            # Thumb
            if hand_landmarks.landmark[thumb_tip].x < hand_landmarks.landmark[thumb_tip - 1].x:
                fingers_up += 1

            # Other fingers
            for tip in finger_tips:
                if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip - 2].y:
                    fingers_up += 1

            # Map finger count to output
            if fingers_up == 0:
                output = "NO"
            elif fingers_up == 1:
                output = " I WANT TO WALK"
            elif fingers_up == 2:
                output = "I NEED REST"
            elif fingers_up == 3:
                output = "I WANT WATER"
            elif fingers_up == 4:
                output = "OK"
            elif fingers_up == 5:
                output = "YES"

    # Print only if changed and 3 seconds passed
    if output and output != last_output and time.time() - last_time > 3:
        print(output)
        last_output = output
        last_time = time.time()

    cv2.imshow("Hand Detection", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()