import cv2

classifier=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')

cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()
    print(frame)
    if not ret:
        break

    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces=classifier.detectMultiScale(gray_frame,scaleFactor=1.1,minNeighbors=5,minSize=(30,30))

    for x,y,w,h in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
    
    cv2.imshow("Face Recognition App : ",frame)

    if cv2.waitKey(1) & 0xFF==ord('q'):
        break

cap.release()
cv2.destroyAllWindows()