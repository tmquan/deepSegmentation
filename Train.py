# -*- coding: utf-8 -*-
"""
Created on Sun Feb 21 10:12:52 2016

@author: tmquan
"""


from Utility 	import *
from Symbol 	import *
from Augment  	import *


logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
# handler = logging.FileHandler('log.txt')
# handler.setLevel(logging.INFO)
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# add the handlers to the logger
# logger.addHandler(handler)

######################################################################################
def augment_img(image, label, shiftedDistance=0, rotatedAngle=1, flip=1, noisy=1):
	# First, transpose image to get normal numpy array
	# image = np.transpose(np.squeeze(image), (1, 2, 0))
	image = np.transpose((image), (1, 2, 0))
	label = np.transpose((label), (1, 2, 0))
	# print image.shape
	
	# Declare random option
	# random_translate = random.randint(-shiftedDistance, shiftedDistance)
	# random_rotatedeg = random.choice(range(-rotatedAngle, rotatedAngle))
	# random_flip      = random.choice([1, 2, 3, 4])
	
	#Random rotate images around center point which is randomly shifted
	if rotatedAngle !=0:
		# random_rotatedeg = random.choice(range(-rotatedAngle, rotatedAngle))
		random_rotatedeg = random.choice([0, 90, 180, 270])
		dimy, dimx, _ = image.shape
		rot_mat = cv2.getRotationMatrix2D(                          \
			(dimx/2+randint(-shiftedDistance, shiftedDistance), 	\
			 dimy/2+randint(-shiftedDistance, shiftedDistance)),    \
			random_rotatedeg, 1.0)
		rotated1 = np.zeros(image.shape)
		rotated2 = np.zeros(image.shape)
		for k in range(image.shape[2]):
			rotated1[:,:,k] = cv2.warpAffine(image[:,:,k], rot_mat, (dimx, dimy))
			rotated2[:,:,k] = cv2.warpAffine(label[:,:,k], rot_mat, (dimx, dimy))
		image = rotated1
		label = rotated2
	image = image.astype(np.float32)
	label = label.astype(np.float32)
	if flip:
		random_flip      = random.choice([1, 2, 3, 4])
		flipped1 = np.zeros(image.shape)
		flipped2 = np.zeros(label.shape)
		for k in range(image.shape[2]):
			if random_flip==1:
				flipped1[:,:,k] = cv2.flip(image[:,:,k], -1)
				flipped2[:,:,k] = cv2.flip(label[:,:,k], -1)
			elif random_flip==2:
				flipped1[:,:,k] = cv2.flip(image[:,:,k], 0)
				flipped2[:,:,k] = cv2.flip(label[:,:,k], 0)
			elif random_flip==3:
				flipped1[:,:,k] = cv2.flip(image[:,:,k], 1)
				flipped2[:,:,k] = cv2.flip(label[:,:,k], 1)
			elif random_flip==4:
				flipped1[:,:,k] = image[:,:,k]
				flipped2[:,:,k] = label[:,:,k]
	else:
		flipped1 = image
		flipped2 = label 
	image = flipped1
	label = flipped2
	image = image.astype(np.float32)
	label = label.astype(np.float32)
		
	noise_typ = "gauss"
	if noisy==1:
		if noise_typ == "gauss":
			row,col,ch= image.shape
			mean = 0
			#var = 0.1
		   #sigma = var**0.5
			gauss = np.random.normal(mean,1,(row,col,ch))
			gauss = gauss.reshape(row,col,ch)
			noisy = image + gauss
			# return noisy
		elif noise_typ == "s&p":
			row,col,ch = image.shape
			s_vs_p = 0.5
			amount = 0.004
			out = image
			# Salt mode
			num_salt = np.ceil(amount * image.size * s_vs_p)
			coords = [np.random.randint(0, i - 1, int(num_salt))
					  for i in image.shape]
			out[coords] = 1

			# Pepper mode
			num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
			coords = [np.random.randint(0, i - 1, int(num_pepper))
					  for i in image.shape]
			out[coords] = 0
			# return out
		elif noise_typ == "poisson":
			vals = len(np.unique(image))
			vals = 2 ** np.ceil(np.log2(vals))
			noisy = np.random.poisson(image * vals) / float(vals)
			# return noisy
		elif noise_typ =="speckle":
			row,col,ch = image.shape
			gauss = np.random.randn(row,col,ch)
			gauss = gauss.reshape(row,col,ch)        
			noisy = image + image * gauss
			# return noisy
		image = noisy
		
	# dd = np.random.randint(3, 11) 
	# sS = np.random.randint(2, 50) 
	# sC = np.random.randint(2, 50)
	# if constrast:
		# image = denoise_tv_chambolle(image, weight=random.uniform(0.001, 0.1))
		# for k in range(image.shape[2]):
			# print image.dtype
			# print image.shape
			# image[:,:,k] = cv2.bilateralFilter(image[:,:,k], 
				# d=dd, 
				# sigmaSpace=sS, 
				# sigmaColor=sC
				# ) 
				
	image = np.transpose(image, (2, 0, 1))
	label = np.transpose(label, (2, 0, 1))
	return image, label 
######################################################################################
def augment_image(
	image,
	label,
	Verbose=False
	):

	image, label = doElastic(image, label)
	image, label = doSquareRotate(image, label)
	image, label = doFlip(image, label)
	image, label = addNoise(image, label)

	return image, label
######################################################################################
def augment_data(X, y):
	progbar = Progbar(X.shape[0])
	for k in range(X.shape[0]):
		image  = np.transpose(X[k], (0, 1, 2))
		label  = np.transpose(y[k], (0, 1, 2))
		
		# print image.shape
		# image, label  = augment_img(image, label)
		image, label = doElastic(image, label, Verbose=False)
		image, label = doSquareRotate(image, label)
		image, label = doFlip(image, label)
		image, label = addNoise(image, label)


		X[k] = image
		y[k] = label 

		# X[k] = np.transpose(image, (2, 0, 1))
		# y[k] = np.transpose(label, (2, 0, 1)) 
		progbar.add(1)
	return X, y 
######################################################################################		
def get_model():
	devs = [mx.gpu(0)]
	network = symmetric_residual()
	# network = get_net_180()
	
	# arg_shape, output_shape, aux_shape = network.infer_shape(data=(1,30,256,256))
	# print "Shape", arg_shape, aux_shape, output_shape
	model = mx.model.FeedForward(ctx=devs,
		symbol          = network,
		num_epoch       = 1,
		learning_rate   = 0.0001,
		wd				= 0.0000000001,
		initializer     = mx.init.Xavier(rnd_type="gaussian", 
							factor_type="in", 
							magnitude=2.34),
		momentum        = 0.8 
		# lr_scheduler=mx.lr_scheduler.FactorScheduler(step=50000, factor=0.1)
		)	
	return model
######################################################################################    
def train():
	X = np.load('X_train.npy')
	y = np.load('y_train.npy')
	
	
	
	X = np.reshape(X, (30, 1, 512, 512))
	y = np.reshape(y, (30, 1, 512, 512))
	
	
	# Preprocess the label
	y = y/255
	# y0 = y
	# y1 = 1-y
	# y  = np.concatenate((y0, y1), axis=1)
	# y  = np.transpose(y)

	y  = y.astype('float32')
	X  = X.astype('float32')
	
	print "Y median", np.median(y)
	print "X shape", X.shape
	print "X dtype", X.dtype
	print "Y shape", y.shape
	print "Y dtype", y.dtype
	
	
	nb_iter = 2001
	epochs_per_iter = 1 
	batch_size =5
	
	model_recon = get_model()
	# dot 		= mx.viz.plot_network(symbol_deconv())
	# print dot
	
	nb_folds = 6
	kfolds = KFold(len(y), nb_folds)
	for i in range(nb_iter):
		print('-'*50)
		print('Iteration {0}/{1}'.format(i + 1, nb_iter))  
		print('-'*50) 
		
		# seed = i #np.random.randint(1, 10e6)
		# Shuffle the data
		print('Shuffle data...')
		seed = np.random.randint(1, 10e6)
		np.random.seed(seed)
		np.random.shuffle(X)
		np.random.seed(seed)
		np.random.shuffle(y)
		
		f = 0
		for train, valid in kfolds:
			print('='*50)
			print('Fold', f+1)
			f += 1
			
			# Extract train, validation set
			X_train = X[train]
			X_valid = X[valid]
			y_train = y[train]
			y_valid = y[valid]
			
			print('Augmenting data for training...')
			X_train, y_train = augment_data(X_train, y_train) # Data augmentation for training 
			X_valid, y_valid = augment_data(X_valid, y_valid) # Data augmentation for training 
			
			# Create two class training
			y0_train = y_train
			y1_train = 1-y_train
			y_train  = np.concatenate((y0_train, y1_train), axis=1)
			
			y0_valid = y_valid
			y1_valid = 1-y_valid
			y_valid  = np.concatenate((y0_valid, y1_valid), axis=1)

			# y_train = np.squeeze(np.reshape(y_train, (25, 1, 1, 1*512*512)))
			# y_valid = np.squeeze(np.reshape(y_valid, (5,  1, 1, 1*512*512)))
			
			
			# print y_train
			# Convert to mxnet type
			X_train    		 = mx.nd.array(X_train)
			X_valid    		 = mx.nd.array(X_valid)
			y_train    		 = mx.nd.array(y_train)
			y_valid    		 = mx.nd.array(y_valid)
			
			# y_train = mx.nd.squeeze(y_train)
			# y_valid = mx.nd.squeeze(y_valid)
			
			print "X_train", X_train.shape
			print "y_train", y_train.shape
			
			
			# prepare data
			data_train = mx.io.NDArrayIter(X_train, y_train,
										   batch_size=batch_size, 
										   shuffle=True, 
										   last_batch_handle='roll_over'
										   )
			data_valid = mx.io.NDArrayIter(X_valid, y_valid,
										   batch_size=batch_size, 
										   shuffle=True, 
										   last_batch_handle='roll_over'
										   )
			
			# network = model_recon.symbol()
			# data_shape = (128,1,512,512)
			# arg_shape, output_shape, aux_shape = network.infer_shape(data=data_shape)
			# print "Shape", arg_shape, aux_shape, output_shape
			# def norm_stat(d):
				# return mx.nd.norm(d)/np.sqrt(d.size)
			# mon = mx.mon.Monitor(100, norm_stat)
			model_recon.fit(X = data_train, 
							# eval_metric = RMSECustom(),
							eval_metric = mx.metric.RMSE(),
							# eval_metric = mx.metric.Accuracy(),
							# eval_metric = mx.metric.CustomMetric(skimage.measure.compare_psnr),
							# eval_metric = mx.metric.MAE(),
							eval_data = data_valid,
							# batch_end_callback = mx.callback.Speedometer(100, 100)
							# monitor=mon 
							)
		if i%10==0:
			model_recon.save('models/model_recon', i)
if __name__ == '__main__':
	train()
