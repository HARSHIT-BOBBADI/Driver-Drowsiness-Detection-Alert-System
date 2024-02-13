import cv2

fire_cascade = cv2.CascadeClassifier("fire_detection.xml")

cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()
    if not ret:
        print("Error reading frame")
        break

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    fire = fire_cascade.detectMultiScale(img, 1.2, 5)

    for (x, y, w, h) in fire:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        print('Fire detected!')

    cv2.imshow('img', img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
