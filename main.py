import requests 
import cv2 
import numpy as np 
import mediapipe as mp
import imutils 

url = "http://192.168.1.4:8080/shot.jpg"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,     
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)
mp_draw = mp.solutions.drawing_utils
  
while True: 
    img_resp = requests.get(url) 
    img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8) 
    img = cv2.imdecode(img_arr, -1) 
    img = imutils.resize(img, width=1000, height=1800) 
    
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(imgRGB)
    
    total_fingers = 0
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            handIndex = result.multi_hand_landmarks.index(hand_landmarks)
            hand_label = result.multi_handedness[handIndex].classification[0].label

            finger_tips = [8, 12, 16, 20]
            thumb_tip = 4
            finger_count = 0

            thumb_tip_x = hand_landmarks.landmark[thumb_tip].x
            thumb_ip_x = hand_landmarks.landmark[thumb_tip-1].x
            
            if (hand_label == "Right" and thumb_tip_x < thumb_ip_x) or \
               (hand_label == "Left" and thumb_tip_x > thumb_ip_x):
                finger_count += 1

            for tip in finger_tips:
                if hand_landmarks.landmark[tip].y < hand_landmarks.landmark[tip-2].y:
                    finger_count += 1

            total_fingers += finger_count
            
            mp_draw.draw_landmarks(
                img, hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_draw.DrawingSpec(color=(255, 200, 87), thickness=2),
            )
            
            

    cv2.putText(img, f"Total numbers: {total_fingers}", 
                (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.imshow("Hand tracking", img) 

    # Press Esc key to exit 
    if cv2.waitKey(1) == 27: 
        break
  
cv2.destroyAllWindows() 



