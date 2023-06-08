import cv2
import os
import matplotlib.pyplot as plt

def convert_bw():
	img_files = os.listdir('im8')
	images = [cv2.imread(f"im8/{img}") for img in img_files]
	images = [cv2.threshold(img, 210, 255, cv2.THRESH_BINARY)[1] for img in images]

	# save images to bw-im8 folder
	for i, img in enumerate(images):
		cv2.imwrite(f"bw-im8/{img_files[i]}", img)

def clean_bw(folder:str='bw-im8'):
	img_files = os.listdir(folder)
	images = [cv2.imread(f"{folder}/{img}", cv2.IMREAD_GRAYSCALE) for img in img_files]

	for img in images:
		img[:,815:] = 0
		img[:,:180] = 0

	# save images to bw-im8 folder
	for i, img in enumerate(images):
		cv2.imwrite(f"{folder}/{img_files[i]}", img)

def edge_detection(folder:str='bw-im8'):
	# Read the original image
	img = cv2.imread(f"{folder}/Estudo08.0116.jpg") 
	# Display original image
	cv2.imshow('Original', img)
	cv2.waitKey(0)

	print(img.shape)
	
	# Convert to graycsale
	img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	# Blur the image for better edge detection
	img_blur = cv2.GaussianBlur(img_gray, (3,3), 0) 
	
	# Sobel Edge Detection
	sobelx = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5) # Sobel Edge Detection on the X axis
	sobely = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5) # Sobel Edge Detection on the Y axis
	sobelxy = cv2.Sobel(src=img_blur, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=5) # Combined X and Y Sobel Edge Detection
	# Display Sobel Edge Detection Images
	cv2.imshow('Sobel X', sobelx)
	cv2.waitKey(0)
	cv2.imshow('Sobel Y', sobely)
	cv2.waitKey(0)
	cv2.imshow('Sobel X Y using Sobel() function', sobelxy)
	cv2.waitKey(0)
	
	# Canny Edge Detection
	edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200) # Canny Edge Detection
	# Display Canny Edge Detection Image
	cv2.imshow('Canny Edge Detection', edges)
	cv2.waitKey(0)
	
	cv2.destroyAllWindows()

def open_hole():
	file_list = os.listdir('defs')
	ix = 600
	iy = 400
	fx = 850
	fy = 600
	for file, i in zip(file_list, range(41)):
		nr = int(file.split('.')[1][-3:])
		print(nr)
		if nr >= 95 and nr <= 115:
			img = cv2.imread(f"defs/{file}", cv2.IMREAD_GRAYSCALE)
			cv2.rectangle(img, (ix, iy), (fx, fy), (0,0,0), -1)
			cv2.imwrite(f"defs-hole/{file}", img)
			if i >= 10 and i < 20:
				ix -= 5
				iy -= 5
				fx += 5
				fy += 5
				# print(ix, iy, fx, fy)
			elif i >= 20 and i < 31:
				ix += 5
				iy += 5
				fx -= 5
				fy -= 5
				# print(ix, iy, fx, fy)

def create_prosthesis():
	file_list_hole = os.listdir('defs-hole')
	file_list = os.listdir('defs')
	for file,file_hole in zip(file_list,file_list_hole):
		nr = int(file.split('.')[1][-3:])
		if nr >= 95 and nr <= 115:
			img = cv2.imread(f"defs/{file}", cv2.IMREAD_GRAYSCALE)
			img_hole = cv2.imread(f"defs-hole/{file_hole}", cv2.IMREAD_GRAYSCALE)
			img_hole = cv2.bitwise_not(img_hole)
			img = cv2.bitwise_and(img, img_hole)
			cv2.imwrite(f"defs-prosthesis/{file}", img)

def test_color(img, value=200):
	"""
	Checks if the pixel is greater than 200
	"""
	return img > value

def change_color_prosthesis():
	"""
	Changes color of the prosthesis to pink
	"""
	import numpy as np
	file_list = os.listdir('defs-prosthesis')
	for file in file_list:
		img = cv2.imread(f"defs-prosthesis/{file}", cv2.IMREAD_GRAYSCALE)
		img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		white_pixels = np.where(np.all(test_color(img), axis=-1))
		img[white_pixels] = [247, 111, 200]
		img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
		cv2.imwrite(f"defs-prosthesis-pink/{file}", img)

def merge_prosthesis_with_hole():
	"""
	Merge images of folder defs-hole with images from folder defs-prosthesis-pink
	"""
	import numpy as np
	file_list_hole = os.listdir('defs-hole')
	file_list_prosthesis = os.listdir('defs-prosthesis')
	for file_hole, file_prosthesis in zip(file_list_hole, file_list_prosthesis):
		nr = int(file_hole.split('.')[1][-3:])
		if nr >= 95 and nr <= 115:
			img_hole = cv2.imread(f"defs-hole/{file_hole}", cv2.IMREAD_GRAYSCALE)
			img_prosthesis = cv2.imread(f"defs-prosthesis-pink/{file_prosthesis}", cv2.IMREAD_GRAYSCALE)
			img_prosthesis = cv2.cvtColor(img_prosthesis, cv2.COLOR_GRAY2BGR)
			img_prosthesis = cv2.cvtColor(img_prosthesis, cv2.COLOR_BGR2RGB)
			white_pixels = np.where(np.all(test_color(img_prosthesis, 50), axis=-1))
			img_prosthesis[white_pixels] = [247, 111, 200]
			img_prosthesis = cv2.cvtColor(img_prosthesis, cv2.COLOR_RGB2BGR)
			img_prosthesis = cv2.cvtColor(img_prosthesis, cv2.COLOR_BGR2GRAY)
			img_prosthesis = cv2.bitwise_not(img_prosthesis)
			img = cv2.bitwise_and(img_hole, img_prosthesis)
			cv2.imwrite(f"defs-prosthesis-hole/{file_prosthesis}", img)
		

def main():
	merge_prosthesis_with_hole()

if __name__ == '__main__':
	main()