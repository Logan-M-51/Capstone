import cv2
import sys
import imageio
from datetime import datetime


count = 0

if __name__ =="__main__":
	capture = cv2.VideoCapture(0)
	capture.set(3,200)
	capture.set(4,200)
	t1 = datetime.now()
	while True:
		t2 = datetime.now()
		delta = t2 - t1
		if (delta.total_seconds() >= 60):
			sys.exit(1)
		success, frame = capture.read()
		cv2.imshow("Image", frame)
		cv2.waitKey(1)

		name = "Traditional_%d.jpg"%count
		frame = cv2.resize(frame, (200,200))
		cv2.imwrite(name, frame)
		count+=1