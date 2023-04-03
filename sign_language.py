import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

finger_tips =[8, 12, 16, 20]
thumb_tip= 4

while True:
    ret,img = cap.read()
    img = cv2.flip(img, 1)
    h,w,c = img.shape
    results = hands.process(img)

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            #accessing the landmarks by their position
            lm_list=[]
            for id ,lm in enumerate(hand_landmark.landmark):
                lm_list.append(lm)

            # Check if thumb is extended
            thumb_extended = lm_list[4].y < lm_list[3].y

            # Check if all other fingers are extended
            fingers_extended = all(lm_list[i].y < lm_list[i-2].y for i in finger_tips)

            # Determine gesture based on finger positions
            if thumb_extended and fingers_extended:
                gesture = "thumbs up"
            else:
                gesture = "thumb down"

            # Draw landmarks and gesture label on image
            mp_draw.draw_landmarks(img, hand_landmark,
            mp_hands.HAND_CONNECTIONS, mp_draw.DrawingSpec((0,0,255),2,2),
            mp_draw.DrawingSpec((0,255,0),4,2))
            cv2.putText(img, gesture, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("hand tracking", img)
    cv2.waitKey(1)
