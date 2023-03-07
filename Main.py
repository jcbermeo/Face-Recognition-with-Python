#   Libraries
import pathlib
import cv2

cascade_path = pathlib.Path(cv2.__file__).parent.absolute() / "data/haarcascade_frontalface_default.xml"
print(cascade_path)

# Clasiffier

clf = cv2.CascadeClassifier(str(cascade_path))

# Line 12 is the input of the camera signal 
camera = cv2.VideoCapture(0)    # Default quantity of cameras

# Line15 is input of a video
camera = cv2.VideoCapture("Videos/People.mp4")

while True:
    _, frame = camera.read();
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = clf.detectMultiScale(
        gray,
        scaleFactor = 1.1,
        minNeighbors = 5,
        minSize=(30,30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    #Plotting a rectangle around the face/faces
    for(x,y,width,height) in faces:
        cv2.rectangle(frame,(x,y), (x+width, y+height), (255,255,0),2)

    cv2.imshow("Faces",frame)
    if cv2.waitKey(1)  == ord("q"):
        break

camera.release()
cv2.destroyAllWindows
