

import cv2
import cvzone

from cvzone.FaceMeshModule import FaceMeshDetector

cap = cv2.VideoCapture(0)

#Finds the face detection and limit to 1 so it doesn't capture other faces
detector = FaceMeshDetector(maxFaces = 1)

while True:
    success, img = cap.read()
    img, faces = detector.findFaceMesh(img, draw = False)
    
    if faces:
        face = faces[0] # will take the first face and will always be true because of maxFaces
        pointLeft = face[145]
        pointRight = face[374]
        
        # #Drawing 
        # cv2.line(img, pointLeft, pointRight, (0,200,0), 3)
        # cv2.circle(img, pointLeft, 5, (255, 0, 255), cv2.FILLED)
        # cv2.circle(img, pointRight, 5, (255, 0, 255), cv2.FILLED)
        
        w, _ = detector.findDistance(pointLeft, pointRight)
        W = 6.3 # the average value of the pupilary distance between men and women will need to find for cats

        
        # # Finding the Focal Length
        # d = 50 
        # f = (w * d) / W
        # print(f)
        
        # Finding distance 
        f = 840 
        d = (W * f) / w
        print(d)
        
        cvzone.putTextRect(img, f"Depth: {int(d)}cm", (face[10][0]-100, face[10][1]-50), scale = 2)

    cv2.imshow("Image", img)
    #The millisecond delay 
    cv2.waitKey(1)