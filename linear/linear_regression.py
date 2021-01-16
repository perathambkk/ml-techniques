"""
Author: Peratham Wiriyathammabhum


"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
eps = np.finfo(float).eps

def linear_regression(Xin, yin, opts, thres=10**-5, max_epochs=200):
	"""
	Perform linear regression on an input row matrix X and a target vector y.
	See: http://cs229.stanford.edu/notes-spring2019/cs229-notes1.pdf
	"""
	X = Xin.copy() # safety
	y = yin.copy()
	y = np.expand_dims(y,axis=1)

	ni, nd = X.shape
	X = np.append(X, np.ones((ni, 1)), axis=1) # append the bias/const term
	
	# theta = weight_init(nd, init_mode='random')
	
	lr = opts['lr']
	bsize = opts['bsize']

	"""
	theta = X^{-1} * y. use a pseudo inverse for X^{-1}.
	Solve the normal equation X'X\theta = X'y.
	"""
	# theta = np.matmul(np.matmul(np.linalg.pinv(np.matmul(X.T, X)), X.T), y)
	theta = np.matmul(np.linalg.pinv(X), y)
	pred = np.matmul(X, theta)
	loss = square_loss(pred, y)
	print('[Info] loss: {:.4f}'.format(float(loss)))
	return theta, pred

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

def square_loss(pred, yin):
	"""
	A typical L2 norm aka. square error.
	"""
	y = yin.copy()
	return np.linalg.norm(pred - y, ord=2)

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
	diabetes = datasets.load_diabetes()
	X, y = diabetes.data, diabetes.target
	theta, pred = linear_regression(X[:, np.newaxis, 2], y, opts, thres=10**-5, max_epochs=max_epochs)
	y = np.expand_dims(y,axis=1)
	# plot
	# input("Press Enter to continue...")
	"""
	plotting data and predictions as well as regression lines.
	"""
	diabetes_X = X[:, np.newaxis, 2]
	idx = np.argsort(diabetes_X, axis=0)
	plt.scatter(diabetes_X[idx].squeeze(1), y[idx].squeeze(1),  color='black')
	plt.plot(diabetes_X[idx].squeeze(1), pred[idx].squeeze(1), color='blue', linewidth=3)

	plt.xticks(())
	plt.yticks(())

	plt.show()
	input("Press Enter to continue...")
	"""
	TODO: plotting model parameters using hinton diagram
	"""
	return

if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser(description='run linear regression.')
	parser.add_argument('--lr', dest='lr',
					  help='learning rate',
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
