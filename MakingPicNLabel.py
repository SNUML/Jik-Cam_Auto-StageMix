import cv2
import numpy as np

print("\n MADE BY SUG, Jik-Cam, SNUML on 2020-03-29.\n")
print("    Labeling Helper for Making Stage Mix DataSet  \n")

####

while True:
	VideoPath = input("Please Type Absolute-Path of your mp4 file. (Exit Key is Ctrl+C.) \nPath (Ex: C:/peekaboo.mp4): ")

	print("\n If it is good timing, press O or o key. Unless, press other keys.  \n")

	Hard = 100  #    Number of Frames

	"""
	= input("How much effort do you put into? (Ex: )\n\nPath: ")	
	"""

	capture = cv2.VideoCapture(VideoPath)

	FrameGap = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) / Hard)


	ret, frame = capture.read()

	Label = np.full((Hard),0,dtype = np.uint8)
	Picture = np.full((Hard ,frame.shape[0],frame.shape[1],frame.shape[2]),0)

	for i in range(1,Hard):

		while(int(capture.get(cv2.CAP_PROP_POS_FRAMES)) != i*FrameGap):
			ret, frame = capture.read()
			if(int(capture.get(cv2.CAP_PROP_POS_FRAMES)) == i*FrameGap):
				cv2.imshow("VideoFrame",frame)		
    
		key = cv2.waitKey(0)
		Picture[i] = frame.copy()
		if (key == ord('o') or key == ord('O')) :
			Label[i] = 1	
	NameIndex = VideoPath.find('.mp4')

	np.save(VideoPath[:NameIndex]+'_pic',Picture)
	np.save(VideoPath[:NameIndex]+'_lbl',Label)
	
	print('Picture array dimension is ',Picture.shape)
	print('Label array dimension is ',Label.shape)
	print('')
	capture.release()
	cv2.destroyAllWindows()
