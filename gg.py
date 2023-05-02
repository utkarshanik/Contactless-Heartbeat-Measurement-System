import cv2
import sys
from tkinter import*

def ge():
	# Webcam Parameters
	webcam = None
	if len(sys.argv) == 2:
		webcam = cv2.VideoCapture(sys.argv[1])
	else:
		webcam = cv2.VideoCapture(0)
	realWidth = 500
	realHeight = 500
	videoWidth = 250
	videoHeight = 250
	videoChannels = 3
	webcam.set(5, realWidth)
	webcam.set(4, realHeight)


	# Output Display Parameters
	font = cv2.FONT_HERSHEY_SIMPLEX
	loadingTextLocation = (20, 40)
	fontScale = 1
	fontColor = (0,0,0)
	lineType = 2
	boxColor = (0, 255, 0)
	boxWeight = 3

	faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
	while True:
		ret, frame = webcam.read()
		if ret == False:
			break
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		faces = faceCascade.detectMultiScale(
			gray,
	
			scaleFactor=1.6,
			minNeighbors=5,     
	   		minSize=(20, 20)
		)

		totalFace = len(faces)		
		detectionFrame = frame[int(videoHeight/2):int(realHeight-videoHeight/2), int(videoWidth/2):int(realWidth-videoWidth/2), :]
		

		outputFrame = detectionFrame 
		outputFrame = cv2.convertScaleAbs(outputFrame)

		frame[int(videoHeight/2):int(realHeight-videoHeight/2), int(videoWidth/2):int(realWidth-videoWidth/2), :] = outputFrame
		cv2.rectangle(frame, (int(videoWidth/6) , int(videoHeight/4)), (int(realWidth-videoWidth/4), int(realHeight-videoHeight/2)), boxColor, boxWeight)
		if(totalFace>0):
				cv2.putText(frame, "Face Found", loadingTextLocation, font, fontScale, fontColor, lineType)
				
		else:
			cv2.putText(frame, "Face Not Found..", loadingTextLocation, font, fontScale, fontColor, lineType)

		imgencode=cv2.imencode('.jpg',frame)[1]
		stringData=imgencode.tostring()
		yield (b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')

	webcam.release()
	cv2.destroyAllWindows()

	