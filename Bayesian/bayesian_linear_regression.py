"""
Author: Peratham Wiriyathammabhum


"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy import stats
eps = np.finfo(float).eps

class BayesLinReg(object):
	"""
	Bayesian linear regression.
	See: Pattern Recognition and Machine Learning by Christopher Bishop ch.3.
		https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/
	Blogs: https://maxhalford.github.io/blog/bayesian-linear-regression/
	"""
	def __init__(self, num_feas, alpha, beta):
		self.num_feas = num_feas
		self.alpha = alpha
		self.beta = beta
		self.mean = np.zeros((num_feas,1))
		self.invcov_mat = np.identity(num_feas) / alpha
		return

	def update(self, x, y):
		"""
		eq 3.50-3.51 in Bishop
		"""
		invcov_mat_n = self.invcov_mat + self.beta * np.outer(x, x)
		mean_n = np.matmul(np.linalg.inv(invcov_mat_n), (np.matmul(self.invcov_mat, self.mean) + self.beta* np.expand_dims(np.dot(y, x), axis=1)))
		assert mean_n.shape == self.mean.shape
		self.mean = mean_n
		self.invcov_mat = invcov_mat_n
		return self

	def predict(self, x):
		"""
		eq 3.58-3.59 in Bishop
		"""
		pred_mean = np.dot(x, self.mean)
		sigma_squared_x = 1./self.beta + np.dot(np.dot(x, np.linalg.inv(self.invcov_mat)), x.T)
		return stats.norm(loc=pred_mean.T, scale=sigma_squared_x ** .5)

	@property
	def weights_dist(self):
		return stats.multivariate_normal(mean=self.mean, cov=np.linalg.inv(self.invcov_mat))

def main(opts):
	from sklearn import metrics

	alpha = opts['alpha']
	beta = opts['beta']
	from sklearn import datasets
	diabetes = datasets.load_diabetes()
	X, y = diabetes.data, diabetes.target
	model = BayesLinReg(num_feas=X.shape[1], alpha=alpha, beta=beta)

	y_pred = np.empty(len(y))

	for i, (xi, yi) in enumerate(zip(X, y)): # one at a time
		y_pred[i] = model.predict(xi).mean()
		model.update(xi, yi)

	print(metrics.mean_absolute_error(y, y_pred))

	# plot
	# input("Press Enter to continue...")


	return

if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser(description='run bayesian linear regression.')
	parser.add_argument('--alpha', dest='alpha',
					  help='alpha',
					  default=.3, type=float)
	parser.add_argument('--beta', dest='beta',
					  help='beta',
					  default=1, type=float)
	args = parser.parse_args()
	opts = vars(args)

	main(opts)
