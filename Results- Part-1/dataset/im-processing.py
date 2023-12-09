import os
import numpy as np
import cv2
import pandas as pd
import urllib



def create_dir(path):
	if not os.path.exists(path):
		os.makedirs(path)


def make_labels(labels):
	lab_num = 0
	feat_num = 0
	for label in labels:
		lab_num += 1
		for label_object in label["objects"]:
			feat_num += 1
			feat_type = label_object["value"]
			label_url = label_object["instanceURI"]
			file_name = f"obj{feat_num}-{feat_type}"
			dir_path = os.path.join(os.cwd(), 'masks')
			label_dir = create_dir(f"{dir_path}/label{lab_num}")
			label_path = label_dir + file_name + '.png'
			urllib.urlretrieve(label_url, label_path)

	mask_num = 0
	for label in os.listdir(os.path.join(os.cwd(), 'masks')):
		mask_num += 1
		file_name = f'label-{mask_num}'
		for feat in os.listdir(label):		
			name, ext = feat.split(".")
			f_num, f_type = name.split("-")
			feat_array = cv2.imread(feat, cv2.IMREAD_GRAYSCALE)
			H, W = feat_array.shape
			label_array = np.zeros((H, W, 3))
			if(f_type == 'ductile'):
				label_array[:, :, 0] = 255*np.logical_or(label_array[:, :, 0], feat_array).astype(int)
			if(f_type == 'brittle'):
				label_array[:, :, 1] = 255*np.logical_or(label_array[:, :, 1], feat_array).astype(int)
			if(f_type == 'background'):
				label_array[:, :, 2] = 255*np.logical_or(label_array[:, :, 2], feat_array).astype(int)

		cv2.imwrite(f'{os.cwd()}/masks/{label}/{file_name}.png', label_array)


def make_images(images):
	for im_num in range(len(images)):
		im_url = images[im_num]
		file_name = f"im{im_num+1}"
		file_path = os.path.join(os.cwd(), 'images')
		image_path = file_path + file_name + '.png'
		urllib.urlretrieve(im_url, image_path)


if __name__ == "__main__":
	data = pd.read_csv(os.path.join(os.pwd(), 'csv (1).csv'))
	labels = data['Label']
	images = data['Labeled Data']
	make_labels(labels)
	make_images(images)





