import os, sys, fileinput, time

results = []

def DeleteSelectedContent():
	path = os.getcwd()
	files = os.listdir(path)
	for f in files:
		curr_path = "/Users/EthanChen1/Desktop/dog_classifier/dogImages/train/" + f 
		if os.path.isdir(curr_path):
			images = os.listdir(curr_path)
			for img in images:
				image_path = curr_path + '/' + img
				print (img)
				print (image_path)
				if os.path.isfile(image_path):
					if '.jpg' in img:
						os.remove(image_path)
DeleteSelectedContent()
