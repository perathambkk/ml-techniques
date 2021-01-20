"""
Author: Peratham Wiriyathammabhum


"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn import metrics
eps = np.finfo(float).eps

def LASSO(Xin, yin, opts, thres=10**-5, max_epochs=200):
	"""
	Perform LASSO (or basis pursuit) on an input row matrix X and a target vector y.
	See: “least absolute shrinkage and selection operator” (Tibshirani 1996)
		 Machine Learning: A Probabilistic Perspective (MLPP) by Kevin Murphy.
		 The Elements of Statistical Learning (ESL) by Trevor Hastie, Robert Tibshirani and Jerome Friedman.
		 http://cazencott.info/dotclear/public/lectures/ma2823_2017/slides/iml_chap7_regularization.pdf
		 http://www.cs.cmu.edu/~pradeepr/convexopt/Lecture_Slides/coordinate_descent.pdf
		 https://courses.cs.washington.edu/courses/cse599c1/13wi/slides/shotgun.pdf
	"""
	X = Xin.copy() # safety
	y = yin.copy()
	y = np.expand_dims(y,axis=1)

	ni, nd = X.shape
	X = np.append(X, np.ones((ni, 1)), axis=1) # append the bias/const term
		
	# lr = opts['lr']
	bsize = opts['bsize']
	lamb_const = opts['lamb_const']
	ridge_lamb_const = opts['ridge_lamb_const']

	"""
	Init theta with ridge regression
	"""
	theta = np.matmul(np.matmul(np.linalg.pinv(np.matmul(X.T, X) + ridge_lamb_const*np.diag(np.ones(nd+1))), X.T), y)
	# theta = weight_init(nd, init_mode='random')
	y_pred = np.matmul(X, theta)
	loss = square_loss(y_pred, y, theta, lamb_const)
	mae = metrics.mean_absolute_error(y, y_pred)
	print('[Info] Init loss: {:.4f} MAE: {:2.2f}'.format(float(loss), float(mae)))
	"""
	Use coordinate descent with subgradient instead of a typical gradient descent.
	T. Wu and K. Lange (2008), “Coordinate descent algorithms for lasso penalized regression”
	Algorithm 13.1 page 441 in Murphy.
	"""
	n_epoch = 0
	while not terminating_cond(n_epoch, max_epochs):
		ind = 0
		loss = 0
		while ind < ni:
			end = ind + bsize if ind + bsize <= ni else ni
			bX = X[ind:end]
			by = y[ind:end]
			pred = np.matmul(bX, theta)
			loss += square_loss(pred, by, theta, lamb_const)

			for j in range(nd + 1): # for each dimension/feature
				a_j = 2.0 * np.linalg.norm(bX[:, j], ord=2) 
				w = theta[j]
				# theta[j] = 0
				# bXi = bX.copy()
				# bXi[:, j] = 0
				y_pred = np.matmul(bX, theta)
				c_j = 2.0 * (np.matmul(bX[:, j].T, by - y_pred + w*bX[:, j])).mean(axis=0)
				c_j /= bsize # This line helps prevent an overflow if batch size > 1.
				if c_j == 0 or a_j == 0:
					theta[j] = w
				else:
					res = soft_thresholding(c_j/a_j, lamb_const/a_j)
					theta[j] = res

			ind += bsize
		n_epoch += 1
		if n_epoch % 1 == 0:
			y_pred = np.matmul(X, theta)
			mae = metrics.mean_absolute_error(y, y_pred)
			print('[Info] epoch: {} loss: {:.4f} MAE: {:2.2f}'.format(n_epoch, float(loss), float(mae)))
	pred = np.matmul(X, theta)
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

def soft_thresholding(a, delta):
	"""
	eq. (13.56) in Murphy.
	"""
	sign_a = np.sign(a)
	res = sign_a*np.maximum(np.abs(a) - delta, 0)
	return res

def hard_thresholding(c_j, a_j, lamb_const):
	"""
	page.434 in Murphy.
	"""
	if np.abs(c_j) < lamb_const:
		res = 0.0
	else:
		res = c_j / a_j 
	return res

def square_loss(pred, yin, theta, lamb_const):
	"""
	A typical L2 norm aka. square error with an L1 regularization term.
	"""
	y = yin.copy()
	return np.linalg.norm(pred - y, ord=2) + lamb_const * np.linalg.norm(theta, ord=1)

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
	# whitening
	X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
	Xin = X
	# Xin = X[:, np.newaxis, 2]
	theta, pred = LASSO(Xin, y, opts, thres=10**-5, max_epochs=max_epochs)
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
	plotting model parameters using hinton diagram
	"""
	from hinton_diagram import hinton
	h_theta = theta.copy() # safety for plotting
	print(h_theta)
	hinton(h_theta)
	plt.show()
	input("Press Enter to continue...")
	plt.close()
	"""
	TODO: plotting the solution path
	"""
	return

if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser(description='run LASSO.')
	# parser.add_argument('--lr', dest='lr',
	# 				  help='learning rate',
	# 				  default=1e-3, type=float)
	parser.add_argument('--bsize', dest='bsize',
					  help='batch size',
					  default=1, type=int)
	parser.add_argument('--lambda', dest='lamb_const',
					  help='lambda for l1',
					  default=1e-3, type=float)
	parser.add_argument('--ridgelambda', dest='ridge_lamb_const',
					  help='lambda for l2',
					  default=1e-3, type=float)
	parser.add_argument('--max_epochs', dest='max_epochs',
					  help='number of iterations to train',
					  default=5, type=int)
	args = parser.parse_args()
	opts = vars(args)

	main(opts)
