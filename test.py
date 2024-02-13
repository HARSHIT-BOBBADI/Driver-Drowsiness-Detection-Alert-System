import cv2
import numpy as np
import dlib
from imutils import face_utils

video_path = 0
cap = cv2.VideoCapture(video_path)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

sleep = 0
active = 0
status = ""
pilot="manual"
AUTO=0
color = (0, 0, 0)
state = 0  # Variable to keep track of consecutive "Padukoku ra sasthav!!!" occurrences

def compute(ptA, ptB):
    dist = np.linalg.norm(ptA - ptB)
    return dist

def blinked(a, b, c, d, e, f):
    up = compute(b, d) + compute(c, e)
    down = compute(a, f)
    ratio = up / (2.0 * down)

    if ratio > 0.25:
        return 2
    elif 0.21 <= ratio <= 0.25:
        return 1
    else:
        return 0

face_frame = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x1 = face.left()
        y1 = face.top()
        x2 = face.right()
        y2 = face.bottom()

        face_frame = frame.copy()
        cv2.rectangle(face_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        landmarks = predictor(gray, face)
        landmarks = face_utils.shape_to_np(landmarks)

        left_blink = blinked(landmarks[36], landmarks[37],
                             landmarks[38], landmarks[41], landmarks[40], landmarks[39])
        right_blink = blinked(landmarks[42], landmarks[43],
                              landmarks[44], landmarks[47], landmarks[46], landmarks[45])
        
        if left_blink == 0 or right_blink == 0:
            sleep += 1
            drowsy = 0
            active = 0
            
            if sleep > 6:
                status = "Padukoku ra sasthav!!!"
                color = (255, 0, 0)
                print(status)
                state += 1  # Increment state variable
                if state == 10:
                    print("Auto Pilot")
                    if not AUTO==3:
                        AUTO+=1
                    else:
                        print("auto activated")
                        pilot="auto pilot"
                       
                    
        
                    state = 0  # Reset state variable to 0

        else:
            drowsy = 0
            sleep = 0
            active += 1
            if active > 6:
                status = "Active :)"
                color = (0, 255, 0)
                print(status)
                state = 0  # Reset state variable to 0

        cv2.putText(frame, status, (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        cv2.putText(frame, pilot, (200, 140), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
        for n in range(0, 68):
            (x, y) = landmarks[n]
            cv2.circle(face_frame, (x, y), 1, (255, 255, 255), -1)

    cv2.imshow("Frame", frame)
    if face_frame is not None:
        cv2.imshow("Result of detector", face_frame)

    key = cv2.waitKey(1)
    if key == ord('q'):  # Press 'q' to quit
        break
    if key== ord('m'):
        pilot="manual"

cap.release()
cv2.destroyAllWindows()
