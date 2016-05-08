# -*- coding: utf-8 -*-
"""
Created on Sat Feb 20 21:03:22 2064

@author: tmquan
"""

from Utility import *


def deconv_factory(data, num_filter, scale, workspace_default=2014):
	deconv = mx.symbol.Deconvolution(data=data, 
		kernel=(2*scale, 2*scale), 
		stride=(scale, scale), 
		pad=(scale/2, scale/2), 
		num_filter=num_filter, 
		no_bias=True, 
		workspace=workspace_default)
	# bn = mx.symbol.BatchNorm(data=deconv)
	act = mx.symbol.Activation(data = deconv, act_type='relu')
	return act
	   
def conv_factory(data, num_filter, kernel, stride, pad, act_type = 'relu', conv_type = 0):
    if conv_type == 0:
        conv = mx.symbol.Convolution(data = data, num_filter = num_filter, kernel = kernel, stride = stride, pad = pad)
        # bn = mx.symbol.BatchNorm(data=conv)
        act = mx.symbol.Activation(data = conv, act_type=act_type)
        return act
    elif conv_type == 1:
        conv = mx.symbol.Convolution(data = data, num_filter = num_filter, kernel = kernel, stride = stride, pad = pad)
        # bn = mx.symbol.BatchNorm(data=conv)
        return conv



def residual_factory(data, num_filter, dim_match):
	if dim_match == True: # if dimension match
		identity_data = data
		conv1 = conv_factory(data=data, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1), act_type='relu', conv_type=0)
		conv2 = conv_factory(data=conv1, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1), conv_type=1)
		# new_data = identity_data + conv2
		new_data = mx.symbol.Concat(*[identity_data, conv2])
		new_data = conv_factory(data=new_data, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1))
		
		act = mx.symbol.Activation(data=new_data, act_type='relu')
		return act
	else:        
		conv1 = conv_factory(data=data, num_filter=num_filter, kernel=(3,3), stride=(2,2), pad=(1,1), act_type='relu', conv_type=0)
		conv2 = conv_factory(data=conv1, num_filter=num_filter, kernel=(3,3), stride=(1,1), pad=(1,1), conv_type=1)
	
		# adopt project method in the paper when dimension increased
		project_data = conv_factory(data=data, num_filter=num_filter, kernel=(1,1), stride=(2,2), pad=(0,0), conv_type=1)
		new_data = project_data + conv2
		act = mx.symbol.Activation(data=new_data, act_type='relu')
		return act

def residual_net(data, n):
    #fisrt 2n layers
    for i in range(n):
        data = residual_factory(data=data, num_filter=128, dim_match=True)
    
    #second 2n layers
    for i in range(n):
        if i==0:
            data = residual_factory(data=data, num_filter=256, dim_match=False)
        else:
            data = residual_factory(data=data, num_filter=256, dim_match=True)
    
    #third 2n layers
    for i in range(n):
        if i==0:
            data = residual_factory(data=data, num_filter=512, dim_match=False)
        else:
            data = residual_factory(data=data, num_filter=512, dim_match=True)
    return data
    
	
def residual_factory(data, num_filter, kernel, stride, pad):	
	identity_data 	= data
	
	conv1 			= mx.symbol.Convolution(data = data, num_filter = num_filter, kernel = kernel, stride = stride, pad = pad)
	# bn1 			= mx.symbol.BatchNorm(data = conv1)
	act1 			= mx.symbol.Activation(data = conv1, act_type='relu')
	
	conv2 			= mx.symbol.Convolution(data = act1, num_filter = num_filter, kernel = kernel, stride = stride, pad = pad)
	# bn2 			= mx.symbol.BatchNorm(data = conv2)
	act2 			= mx.symbol.Activation(data = conv2, act_type='relu')
	
	conv3 			= mx.symbol.Convolution(data = act2, num_filter = num_filter, kernel = kernel, stride = stride, pad = pad)
	new_data 		= conv3+identity_data
	# new_data 		= mx.symbol.Concat(*[conv3, identity_data])
	# bn3 			= mx.symbol.BatchNorm(data = new_data)
	act3 			= mx.symbol.Activation(data = new_data, act_type='relu')
	
	return act3

def downsample_factory(data, n):
	# conv 			= mx.symbol.Convolution(data 		= data, 
											# num_filter 	= 	128,
											# kernel		=	(15,15),
											# stride 		=	(1,1),
											# pad			=	(7,7))
											
	# act 			= mx.symbol.Activation(data = conv, act_type='relu')
	down_group 		= data
	for i in range(n):
		conv_group = residual_factory(data 			= 	down_group, 
									  num_filter 	= 	128,
									  kernel		=	(3,3),
									  stride 		=	(1,1),
									  pad 			=	(1,1))
		conv_group = residual_factory(data 			= 	conv_group, 
									  num_filter 	= 	128,
									  kernel		=	(3,3),
									  stride 		=	(1,1),
									  pad 			=	(1,1))
		conv_group = residual_factory(data 			= 	conv_group, 
									  num_filter 	= 	128,
									  kernel		=	(3,3),
									  stride 		=	(1,1),
									  pad 			=	(1,1))	
		down_group = mx.symbol.Pooling(data			= conv_group,
									   kernel 		= (2,2),
									   stride 		= (2,2),
									   pool_type 	= 'max')
	return down_group

def upsample_factory(data, n):
	
	conv_group = data
	for i in range(n):
		# down_group = mx.symbol.Pooling(data			= conv_group,
									   # kernel 		= (2,2),
									   # stride 		= (2,2),
									   # pool_type 	= 'max')
		scale = 2
		workspace_default = 1024
		deconv = mx.symbol.Deconvolution(data		=	conv_group, 
										 kernel		=	(2*scale, 2*scale), 
										 stride		=	(scale, scale), 
										 pad		=	(scale/2, scale/2), 
										 num_filter	=	128, 
										 no_bias	=	True, 
										 workspace	=	workspace_default)
										 
		conv_group = residual_factory(data 			= 	deconv, 
									  num_filter 	= 	128,
									  kernel		=	(3,3),
									  stride 		=	(1,1),
									  pad 			=	(1,1))
		conv_group = residual_factory(data 			= 	conv_group, 
									  num_filter 	= 	128,
									  kernel		=	(3,3),
									  stride 		=	(1,1),
									  pad 			=	(1,1))
		conv_group = residual_factory(data 			= 	conv_group, 
									  num_filter 	= 	128,
									  kernel		=	(3,3),
									  stride 		=	(1,1),
									  pad 			=	(1,1))
									  

	return conv_group
def inception_factory(data, num_filter, kernel=(1,1), stride=(1,1), pad=(0,0)):	
	conv 			= mx.symbol.Convolution(data = data, num_filter = num_filter, kernel = kernel, stride = stride, pad = pad)
	bn  			= mx.symbol.BatchNorm(data = conv)
	act 			= mx.symbol.Activation(data = bn, act_type='relu')
	return act

def symmetric_residual():
	
	
	data = mx.symbol.Variable('data')
	data = data/255
	
	# Before down
	conv0a = conv_factory(
		data		= 	data, 
		num_filter	=	64, 
		kernel		=	(3,3), 
		stride		=	(1,1), 
		pad			= 	(1,1), 
		act_type = 'relu')
	
	
	####################################################################
	scale = 2
	workspace_default = 1024	
	####################################################################
	# Residual with concat is here
	conv1a = conv_factory(
		data		= 	conv0a, 
		num_filter	=	64, 
		kernel		=	(3,3), 
		stride		=	(1,1), 
		pad			= 	(1,1), 
		act_type = 'relu')
		
	res1a = residual_factory(
		data 		= 	conv1a,
		num_filter	=	64, 
		kernel		=	(3,3), 
		stride		=	(1,1), 
		pad			= 	(1,1))
	
	down1a = mx.symbol.Pooling(
		data		= res1a,
		kernel 		= (2,2),
		stride 		= (2,2),
		pool_type 	= 'max')
	####################################################################	
	conv2a = conv_factory(
		data		= 	down1a, 
		num_filter	=	32, 
		kernel		=	(3,3), 
		stride		=	(1,1), 
		pad			= 	(1,1), 
		act_type = 'relu')
	res2a = residual_factory(
		data 		= 	conv2a,
		num_filter	=	32, 
		kernel		=	(3,3), 
		stride		=	(1,1), 
		pad			= 	(1,1))
	down2a = mx.symbol.Pooling(
		data		= res2a,
		kernel 		= (2,2),
		stride 		= (2,2),
		pool_type 	= 'max')
	####################################################################
	conv3a = conv_factory(
		data		= 	down2a, 
		num_filter	=	64, 
		kernel		=	(3,3), 
		stride		=	(1,1), 
		pad			= 	(1,1), 
		act_type = 'relu')
	res3a = residual_factory(
		data 		= 	conv3a,
		num_filter	=	64, 
		kernel		=	(3,3), 
		stride		=	(1,1), 
		pad			= 	(1,1))
	down3a = mx.symbol.Pooling(
		data		= res3a,
		kernel 		= (2,2),
		stride 		= (2,2),
		pool_type 	= 'max')
	####################################################################
	conv4a = conv_factory(
		data		= 	down3a, 
		num_filter	=	96, 
		kernel		=	(3,3), 
		stride		=	(1,1), 
		pad			= 	(1,1), 
		act_type = 'relu')
	res4a = residual_factory(
		data 		= 	conv4a,
		num_filter	=	96, 
		kernel		=	(3,3), 
		stride		=	(1,1), 
		pad			= 	(1,1))
	down4a = mx.symbol.Pooling(
		data		= res4a,
		kernel 		= (2,2),
		stride 		= (2,2),
		pool_type 	= 'max')
	####################################################################
	conv_mid = conv_factory(
		data		= 	down4a, 
		num_filter	=	128, 
		kernel		=	(3,3), 
		stride		=	(1,1), 
		pad			= 	(1,1), 
		act_type = 'relu')
	####################################################################
	up4b = mx.symbol.Deconvolution(
		data		=	conv_mid, 
		kernel		=	(2*scale, 2*scale), 
		stride		=	(scale, scale), 
		pad			=	(scale/2, scale/2), 
		num_filter	=	96, 
		no_bias		=	True, 
		workspace	=	workspace_default)
	
	# ccat4b = mx.symbol.Concat(*[up4b, res4a])
	ccat4b = up4b + res4a
	ccat4b = conv_factory(
		data		= 	ccat4b, 
		num_filter	=	96, 
		kernel		=	(3,3), 
		stride		=	(1,1), 
		pad			= 	(1,1), 
		act_type = 'relu')
		
	res4b = residual_factory(
		data 		= 	ccat4b,
		num_filter	=	96, 
		kernel		=	(3,3), 
		stride		=	(1,1), 
		pad			= 	(1,1))
	conv4b = conv_factory(
		data		= 	res4b, 
		num_filter	=	64, 
		kernel		=	(3,3), 
		stride		=	(1,1), 
		pad			= 	(1,1), 
		act_type = 'relu')
	####################################################################
	up3b = mx.symbol.Deconvolution(
		data		=	conv4b, 
		kernel		=	(2*scale, 2*scale), 
		stride		=	(scale, scale), 
		pad			=	(scale/2, scale/2), 
		num_filter	=	64, 
		no_bias		=	True, 
		workspace	=	workspace_default)
	
	# ccat3b = mx.symbol.Concat(*[up3b, res3a])
	ccat3b = up3b + res3a
	ccat3b = conv_factory(
		data		= 	ccat3b, 
		num_filter	=	64, 
		kernel		=	(3,3), 
		stride		=	(1,1), 
		pad			= 	(1,1), 
		act_type = 'relu')
		
	res3b = residual_factory(
		data 		= 	ccat3b,
		num_filter	=	64, 
		kernel		=	(3,3), 
		stride		=	(1,1), 
		pad			= 	(1,1))
	conv3b = conv_factory(
		data		= 	res3b, 
		num_filter	=	32, 
		kernel		=	(3,3), 
		stride		=	(1,1), 
		pad			= 	(1,1), 
		act_type = 'relu')
	####################################################################
	up2b = mx.symbol.Deconvolution(
		data		=	conv3b, 
		kernel		=	(2*scale, 2*scale), 
		stride		=	(scale, scale), 
		pad			=	(scale/2, scale/2), 
		num_filter	=	32, 
		no_bias		=	True, 
		workspace	=	workspace_default)
	
	# ccat2b = mx.symbol.Concat(*[up2b, res2a])
	ccat2b = up2b + res2a
	ccat2b = conv_factory(
		data		= 	ccat2b, 
		num_filter	=	32, 
		kernel		=	(3,3), 
		stride		=	(1,1), 
		pad			= 	(1,1), 
		act_type = 'relu')
		
	res2b = residual_factory(
		data 		= 	ccat2b,
		num_filter	=	32, 
		kernel		=	(3,3), 
		stride		=	(1,1), 
		pad			= 	(1,1))
	conv2b = conv_factory(
		data		= 	res2b, 
		num_filter	=	64, 
		kernel		=	(3,3), 
		stride		=	(1,1), 
		pad			= 	(1,1), 
		act_type = 'relu')
	####################################################################
	up1b = mx.symbol.Deconvolution(
		data		=	conv2b, 
		kernel		=	(2*scale, 2*scale), 
		stride		=	(scale, scale), 
		pad			=	(scale/2, scale/2), 
		num_filter	=	64, 
		no_bias	=	True, 
		workspace	=	workspace_default)
	
	# ccat1b = mx.symbol.Concat(*[up1b, res1a])
	ccat1b = up1b + res1a
	ccat1b = conv_factory(
		data		= 	ccat1b, 
		num_filter	=	64, 
		kernel		=	(3,3), 
		stride		=	(1,1), 
		pad			= 	(1,1), 
		act_type = 'relu')
		
	res1b = residual_factory(
		data 		= 	ccat1b,
		num_filter	=	64, 
		kernel		=	(3,3), 
		stride		=	(1,1), 
		pad			= 	(1,1))
	conv1b = conv_factory(
		data		= 	res1b, 
		num_filter	=	64, 
		kernel		=	(3,3), 
		stride		=	(1,1), 
		pad			= 	(1,1), 
		act_type = 'relu')
	# ####################################################################
	# out = conv1b
	# numclass = 2
	# # After up
	# # Inception unit for segmentation
	# conv0b = conv_factory(
		# data		= 	out, 
		# num_filter	=	numclass, 
		# kernel		=	(1,1), 
		# stride		=	(1,1), 
		# pad			= 	(0,0), 
		# act_type = 'relu')
	
	####################################################################
	scale = 16
	up4c = mx.symbol.Deconvolution(
		data        =   conv_mid, 
		kernel      =   (2*scale, 2*scale), 
		stride      =   (scale, scale), 
		pad         =   (scale/2, scale/2), 
		num_filter  =   96, 
		no_bias     =   True, 
		workspace   =   workspace_default)
	
	scale = 8
	up3c = mx.symbol.Deconvolution(
		data        =   conv4b, 
		kernel      =   (2*scale, 2*scale), 
		stride      =   (scale, scale), 
		pad         =   (scale/2, scale/2), 
		num_filter  =   96, 
		no_bias     =   True, 
		workspace   =   workspace_default)
	scale = 4
	up2c = mx.symbol.Deconvolution(
		data        =   conv3b, 
		kernel      =   (2*scale, 2*scale), 
		stride      =   (scale, scale), 
		pad         =   (scale/2, scale/2), 
		num_filter  =   96, 
		no_bias     =   True, 
		workspace   =   workspace_default)
	scale = 2
	up1c = mx.symbol.Deconvolution(
		data        =   conv2b, 
		kernel      =   (2*scale, 2*scale), 
		stride      =   (scale, scale), 
		pad         =   (scale/2, scale/2), 
		num_filter  =   96, 
		no_bias     =   True, 
		workspace   =   workspace_default)
	# fusion = mx.symbol.Concat(*[conv1b, up4c, up3c, up2c, up1c])
	fusion = conv1b + up4c + up3c + up2c + up1c
	####################################################################
	# out = conv1b
	out = fusion
	numclass = 2
	# After up
	# Inception unit for segmentation
	conv0b = conv_factory(
		data		= 	out, 
		num_filter	=	numclass, 
		kernel		=	(1,1), 
		stride		=	(1,1), 
		pad			= 	(0,0), 
		act_type = 'relu')
	####################################################################
	
	# bn = mx.symbol.BatchNorm(data=conv0b)
	# act = mx.symbol.Activation(data = bn, act_type='relu')
	sm   = mx.symbol.LogisticRegressionOutput(data=conv0b, name="softmax")
	return sm
if __name__ == '__main__':
	# Draw the net

	data 	= mx.symbol.Variable('data')
	network = symmetric_residual()
	dot = mx.viz.plot_network(network,
		None
		# shape={"data" : (128, 1, 256, 256)}
		) 
	dot.graph_attr['rankdir'] = 'RL'
	
	
	
	