import cv2 
import sys
import os
import mediapipe as mp
import numpy as np
import time
from datetime import datetime
import json

data = []
height, width = 200, 200


def parse_landmark(data):
	ret = []
	cords = str(data).split("\n")[0:3]
	for cord in cords:
		ret.append(float(cord.split(":")[1]))
	return ret


if __name__ =="__main__":

	cap = cv2.VideoCapture(0)
	#Set the Webcam 
	cap.set(3,200)
	cap.set(4,200)
	mpHands = mp.solutions.hands
	hands = mpHands.Hands()
	mpDraw = mp.solutions.drawing_utils
	threads = list()
	prev = 0
	frameCnt = 0
	count = 0

	t1 = datetime.now()
	# Open Camera
	while (cap.isOpened()):
		t2 = datetime.now()
		delta = t2 - t1
		if (delta.total_seconds() >= 45):
			break
		try:
			success, img = cap.read()
			imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			results = hands.process(imgRGB)

			# Check for existing model

			if results.multi_hand_landmarks:
				for hand in results.multi_hand_landmarks:
					saved_frame = np.zeros((height,width,3), np.uint8)
					mpDraw.draw_landmarks(saved_frame, hand, mpHands.HAND_CONNECTIONS)
					mpDraw.draw_landmarks(img, hand, mpHands.HAND_CONNECTIONS)
					name = "Landmark_C_%d.jpg"%count
					cv2.imwrite(name, saved_frame)
					count += 1

			current = time.time()
			fps = 1 / (current - prev)
			prev = current
			cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,(255, 0, 255), 3)
			cv2.imshow("Image", img)
			cv2.waitKey(1)     
		except KeyboardInterrupt:
			print(data)
			break



