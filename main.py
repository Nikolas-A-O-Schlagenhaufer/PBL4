import cv2
import os

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
	import numpy as np
	file_list_hole = os.listdir('defs-hole')
	file_list = os.listdir('defs')
	ix = 600
	iy = 400
	fx = 850
	fy = 600
	for file,file_hole,i in zip(file_list,file_list_hole,range(41)):
		nr = int(file.split('.')[1][-3:])
		print(nr)
		if nr >= 95 and nr <= 115:
			# img = cv2.imread(f"defs/{file}", cv2.IMREAD_GRAYSCALE)
			img_hole = cv2.imread(f"defs-hole/{file_hole}", cv2.IMREAD_GRAYSCALE)
			img_hole_flip = cv2.flip(img_hole, 1)
			result = np.zeros(img_hole.shape, dtype=np.uint8)
			result[iy:fy,ix:fx] = img_hole_flip[iy:fy,ix:fx]

			cv2.imwrite(f"defs-prosthesis/{file}", result)

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

def merge_prosthesis_with_hole():
	"""
	Merge images of folder defs-hole with images from folder defs-prosthesis-pink
	"""
	file_list_hole = os.listdir('defs-hole')
	file_list_prosthesis = os.listdir('defs-prosthesis')
	for file_hole, file_prosthesis in zip(file_list_hole, file_list_prosthesis):
		nr = int(file_hole.split('.')[1][-3:])
		if nr >= 95 and nr <= 115:
			print(nr)
			img_hole = cv2.imread(f"defs-hole/{file_hole}", cv2.IMREAD_GRAYSCALE)
			img_prosthesis = cv2.imread(f"defs-prosthesis-pink/{file_prosthesis}", cv2.IMREAD_GRAYSCALE)
			img_hole = cv2.threshold(img_hole, 127, 255, cv2.THRESH_BINARY)[1]
			img_prosthesis = cv2.threshold(img_prosthesis, 127, 255, cv2.THRESH_BINARY)[1]
			img_merged = cv2.bitwise_or(img_hole, img_prosthesis)
			cv2.imwrite(f"defs-prosthesis-merge/{file_prosthesis}", img_merged)
		

def main():
	# convert_bw()
	# clean_bw()
	# open_hole()
	# create_prosthesis()
	merge_prosthesis_with_hole()

if __name__ == '__main__':
	main()