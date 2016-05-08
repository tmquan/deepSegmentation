from Utility import *

def arr2img(imageFile, img):
	"""
	Read images tif files and store as numpy array
	"""
	# img = skimage.io.imread(imageFile)
	print "File name: ", imageFile
	print "Shape    : ", img.shape
	skimage.io.imsave(imageFile, img)
	return 
	# return img
	
def deploy():
	X = np.load('X_train.npy')
	y = np.load('y_train.npy')
	#X = X/255
	#y = y/255
	# X = np.reshape(X, (128, 1, 128, 128))
	# y = np.reshape(y, (128, 1, 128, 128))	
	# X = np.reshape(X, (128, 1, 256, 256))
	# y = np.reshape(y, (128, 1, 256, 256)) 
	X = np.reshape(X, (30, 1, 512, 512))
	y = np.reshape(y, (30, 1, 512, 512))
	
	print "X.shape", X.shape
	print "y.shape", y.shape
	# X_deploy = X
	# X_deploy = X[26:27,:,:,:]
	# X_deploy = X[49:50,:,:,:]
	# y_deploy = y[49:50,:,:,:]
	# X_deploy = X[51:52,:,:,:]
	# y_deploy = y[49:50,:,:,:]
	X_deploy = X[1:2,:,:,:]
	y_deploy = y[21:22,:,:,:]
	
	print "X_deploy.shape", X_deploy.shape
	# print "y_deploy.shape", y_deploy.shape
	# Load model
	iter =700
	model_recon 	= mx.model.FeedForward.load('models/model_recon', iter, ctx=mx.gpu(0))
	# model_recon 	= mx.model.FeedForward.load('models/model_recon', iter)
	
	network = model_recon.symbol()
	arg_shape, output_shape, aux_shape = network.infer_shape(data=(1,2,512,512))
	print "Shape", arg_shape, aux_shape, output_shape
	
	# Perform prediction
	# batch_size = 1
	print('Predicting on data...')
	pred_recon  = model_recon.predict(X_deploy, num_batch=None)
	
	# X_deploy  = np.array(X_deploy)
	# X_deploy  = np.reshape(X_deploy, (512, 512))
	# plt.imshow((np.absolute(X_deploy)) , cmap = plt.get_cmap('gray'))
	# plt.show()	
	

    
	#print pred_recon
	#print X_deploy
	print "pred_recon.shape", pred_recon.shape
	pred_recon  = np.reshape(pred_recon, (2,512, 512))
	pred_recon  = np.array(pred_recon[0,:,:])
	
	# # Post processing
	# strel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
	# strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
	# strel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
	# pred_recon  *= 255
	# pred_recon  = pred_recon.astype(np.uint8)
	
	# pred_recon  = cv2.erode(pred_recon,strel,iterations = 1)
	# pred_recon  = cv2.morphologyEx(pred_recon, cv2.MORPH_OPEN, strel,iterations = 2)
	
	plt.imshow(np.hstack( (
							np.squeeze(X_deploy), 
							# np.squeeze(y_deploy), 
							#np.squeeze(y_deploy),
							255*(pred_recon),
							
						  ), 
						  ) , cmap = plt.get_cmap('gray'))
	plt.show()	
	
	# # Save stack
	# X_deploy = X
	# pred_recon  = model_recon.predict(X_deploy, num_batch=None)
 	# pred_recon  *= 255
	# pred_recon  = pred_recon.astype(np.uint8)
	# for k in range(pred_recon.shape[0]):
		# tmp = np.squeeze(pred_recon[k,:,:,:])
		# tmp = cv2.erode(tmp,strel,iterations = 1)
		# tmp = cv2.morphologyEx(tmp, cv2.MORPH_OPEN, strel,iterations = 2)
		# pred_recon[k,:,:,:] = tmp
	# arr2img('deploy.tif', pred_recon)
	
if __name__ == '__main__':
	deploy()
