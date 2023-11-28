

import cv2
import cvzone

from cvzone.FaceMeshModule import FaceMeshDetector

import numpy as np

cap = cv2.VideoCapture(0)

#Finds the face detection and limit to 1 so it doesn't capture other faces
detector = FaceMeshDetector(maxFaces = 1)


textList = ["Welcome to", "Your Workshop", "Beta testing if it", "Works"]

while True:
    success, img = cap.read()
    imgText = np.zeros_like(img)
    img, faces = detector.findFaceMesh(img, draw = False)
    
    if faces:
        face = faces[0] # will take the first face and will always be true because of maxFaces
        pointLeft = face[145]
        pointRight = face[374]
        
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
        
        for i, text in enumerate(textList):
            singleHeight = 50
            cv2.putText(img, text, (50, 50+(i * singleHeight)), 1, (255, 255, 255), 2)

    imgStacked = cvzone.stackImages([img, imgText], 2, 1)
    cv2.imshow("Image", imgStacked)
    #The millisecond delay 
    cv2.waitKey(1)