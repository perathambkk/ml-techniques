"""
Author: Peratham Wiriyathammabhum


"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
eps = np.finfo(float).eps

def logistic_regression(Xin, yin, opts, thres=10**-5, max_epochs=200):
	"""
	Perform logistic regression on an input row matrix X and a target vector y.
	See: http://www.ciml.info/dl/v0_9/ciml-v0_9-ch06.pdf
	"""
	X = Xin.copy() # safety
	y = yin.copy()
	y = np.expand_dims(y,axis=1)

	ni, nd = X.shape
	X = np.append(X, np.ones((ni, 1)), axis=1) # append the bias/const term
	
	theta = weight_init(nd, init_mode='random')
	
	lr = opts['lr']
	bsize = opts['bsize']
	lamb_const = opts['lamb_const']

	n_epoch = 0
	while not terminating_cond(n_epoch, max_epochs):
		ind = 0
		loss = 0
		while ind < ni:
			end = ind + bsize if ind + bsize <= ni else ni
			bX = X[ind:end]
			by = y[ind:end]
			gz = sigmoid(np.matmul(bX, theta))
			pred = gz
			loss += cross_entropy_loss(pred, by) + (lamb_const / 2.0) * np.linalg.norm(theta, ord=2)
			grad = (np.matmul(bX.T, (pred - by))).sum(axis=1)
			grad = np.expand_dims(grad, 1)
			theta -= lr * grad + lamb_const * theta # sgd
			ind += bsize
		n_epoch += 1
		if n_epoch % 10 == 0:
			acc = compute_accuracy(X, y, theta)
			print('[Info] epoch: {} loss: {:.4f} acc: {:2.2f}%'.format(n_epoch, float(loss), float(acc)))
	return theta

def compute_accuracy(X, yin, theta):
	y = yin.copy()
	gz = sigmoid(np.matmul(X,theta))
	acc = (gz == y).mean()
	return acc*100.

def weight_init(nd, init_mode='random'):
	if init_mode == 'random':
		theta = np.random.randn(nd + 1, 1) # plus the bias/const term
	elif init_mode=='xavier':
		"""
		Glorot, Xavier, and Yoshua Bengio. 
		"Understanding the difficulty of training deep feedforward neural networks." 
		Proceedings of the thirteenth international conference on artificial 
		intelligence and statistics. 2010.
		"""
		pass
	else:
		pass
	return theta

def cross_entropy_loss(pred, yin):
	"""
	https://en.wikipedia.org/wiki/Cross_entropy#Cross-entropy_loss_function_and_logistic_regression
	"""
	# n = y.shape[0]
	# loss = -(y*np.log(pred+eps) + (1-y)*np.log(1-pred+eps))/n
	# loss = loss.sum(axis=0)
	# return loss
	"""
	for numerical stability use log sum:
		??max_val = (-input).clamp(min=0)
		??loss = input - input * target + max_val + ((-max_val).exp() + (-input - max_val).exp()).log()
		https://discuss.pytorch.org/t/numerical-stability-of-bcewithlogitsloss/8246
		http://tagkopouloslab.ucdavis.edu/?p=2197
		https://www.xarg.org/2016/06/the-log-sum-exp-trick-in-machine-learning/
	"""
	y = yin.copy()
	n = y.shape[0]
	max_value = np.clip(pred, eps, 1-eps)
	loss = -(y * np.log(max_value) + (1-y) * np.log(1-max_value))/n
	loss = loss.sum(axis=0)
	return loss

def _pos_sigmoid(x):
	z = np.exp(-x)
	return 1 / (1 + z)

def _neg_sigmoid(x):
	z = np.exp(x)
	return z / (1 + z)

def sigmoid(x):
	#return 1.0 / (1.0 + np.exp(-x))
	"""
	Numerically stable sigmoid function.
		see: https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
	"""
	sx = x.copy()
	pos_ind = sx >= 0
	neg_ind = ~pos_ind

	sx[pos_ind] = _pos_sigmoid(sx[pos_ind])
	sx[neg_ind] = _neg_sigmoid(sx[neg_ind])
	return sx

def terminating_cond(nite, max_iters):
	if nite >= max_iters:
		print('[Info] terminate at iter: {}'.format(nite))
		return True
	# elif np.linalg.norm(mu - nmu) < thres:
	#   print('[Info] terminate at iter: {}'.format(nite))
	#   return True
	else:
		return False

def main(opts):
	max_epochs = opts['max_epochs']
	from sklearn import datasets
	breast_cancer = datasets.load_breast_cancer()
	X, y = breast_cancer.data, breast_cancer.target
	theta = logistic_regression(X, y, opts, thres=10**-5, max_epochs=max_epochs)

	# plot
	# input("Press Enter to continue...")
	"""

	"""
	return

if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser(description='run logistic regression with l2 regularization.')
	parser.add_argument('--lr', dest='lr',
					  help='learning rate',
					  default=1e-3, type=float)
	parser.add_argument('--lambda', dest='lamb_const',
					  help='lambda for l2',
					  default=1e-3, type=float)
	parser.add_argument('--bsize', dest='bsize',
					  help='batch size',
					  default=16, type=int)
	parser.add_argument('--max_epochs', dest='max_epochs',
					  help='number of iterations to train',
					  default=200, type=int)
	args = parser.parse_args()
	opts = vars(args)

	main(opts)
