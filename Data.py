from Utility import *
from Augment import *

train_volume_file = "data/train-volume.tif"
train_labels_file = "data/train-labels.tif"
test_volume_file  = "data/test-volume.tif"

def img2arr(imageFile):
	"""
	Read images tif files and store as np array
	"""
	img = skimage.io.imread(imageFile)
	print "File name: ", imageFile
	print "Shape    : ", img.shape
	return img

def data():
	train_image = img2arr(train_volume_file)
	train_label = img2arr(train_labels_file)
	test_image  = img2arr(test_volume_file)
	
	
	
if __name__ == '__main__': 
	train_image = img2arr(train_volume_file)
	train_label = img2arr(train_labels_file)

	# # Test this image
	# image = train_image[0,:,:]
	# label = train_label[0,:,:]
	# image, label = elastic(image, label, Verbose=True)