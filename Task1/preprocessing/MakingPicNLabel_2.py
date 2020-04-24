import cv2
import numpy as np
from matplotlib import pyplot as plt


print("\n MADE BY SUG, Jik-Cam, SNUML on 2020-04-24.\n")
print("    Preprocessing Helper version2 for Making Stage Mix DataSet  \n")

####

WorkingString = 'I\'m working...'

face_cascade = cv2.CascadeClassifier(
	'./haarcascade_frontalface_default.xml'
)

while True:
	VideoPath = input("Please Type Absolute-Path of your mp4 file. (Exit Key is Ctrl+C.) \nPath (Ex: C:/peekaboo.mp4): ")

	Hard = 300  #    Number of Frames

	capture = cv2.VideoCapture(VideoPath)

	FrameGap = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) / Hard)
	print(FrameGap)
	ret, frame = capture.read()

	print(frame.shape)

	Label = np.full((Hard),0,dtype = np.uint8)
	Picture = np.full((Hard ,frame.shape[0],frame.shape[1],frame.shape[2]),0,dtype = np.float32)

	for i in range(1,Hard):
		while(int(capture.get(cv2.CAP_PROP_POS_FRAMES)) != i*FrameGap):
			ret, frame = capture.read()		

		Picture[i] = frame.copy()
		grayImage = cv2.cvtColor(Picture[i],cv2.COLOR_BGR2GRAY)

		x_ori = grayImage.shape[1]

		grayImage = np.array(grayImage, dtype='uint8')

		faces = face_cascade.detectMultiScale(grayImage,1.03,5)
		flag = False
		for (x,y,w,h) in faces:
			if (x>= x_ori/4 and x+w <= x_ori/4*3 and w>=200 and h>=200):
				flag = True
		
		if (flag):
			Label[i] = 1
			#
			# The lower is for sophisticated user. If you erase lower ''' and make upper line ''',
			#  You can make more accurate data. But it may take about 15~20 minutes.
			#
			'''
			cv2.imshow("VideoFrame",frame)	
			key = cv2.waitKey(0)
			if (key == ord('o') or key == ord('O')) :
				Label[i] = 1	
			'''
		
		if (i%10==0):
			print(WorkingString,i,'/',Hard)
	
	NameIndex = VideoPath.find('.mp4')
	
	np.save(VideoPath[:NameIndex]+'_pic',Picture)
	np.save(VideoPath[:NameIndex]+'_lbl',Label)
		
	print('Picture array dimension is ',Picture.shape)
	print('Label array dimension is ',Label.shape)
	print('')
	capture.release()
	cv2.destroyAllWindows()