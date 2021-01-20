"""
Author: Peratham Wiriyathammabhum


"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
eps = np.finfo(float).eps

def linear_regression(Xin, yin, opts):
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

def main(opts):
	from sklearn import datasets
	diabetes = datasets.load_diabetes()
	X, y = diabetes.data, diabetes.target
	# whitening
	X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

	theta, pred = linear_regression(X[:, np.newaxis, 2], y, opts)
	# theta, pred = linear_regression(X, y, opts)
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
	plt.close()
	"""
	TODO: plotting model parameters using hinton diagram
	"""
	from hinton_diagram import hinton
	h_theta = theta.copy() # safety for plotting
	# h_theta = (h_theta - np.mean(h_theta, axis=0)) / np.std(h_theta, axis=0)
	# h_theta = np.squeeze(np.stack((h_theta, h_theta)))
	hinton(h_theta)
	plt.show()
	input("Press Enter to continue...")
	plt.close()
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
	args = parser.parse_args()
	opts = vars(args)

	main(opts)
