"""
Author: Peratham Wiriyathammabhum


"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
# import scipy.sparse.linalg as linalg
from numpy import linalg as LA
from graph_utils import affinity_graph

def classical_mds(Xin):
	"""
	Perform classical Multidimensional Scaling (classical MDS) on an input row matrix X.
	
	See: https://en.wikipedia.org/wiki/Multidimensional_scaling
	"""
	X = Xin.copy() # for safety
	ni, nd = X.shape
	# X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

	# Set up the squared proximity matrix
	D = affinity_graph(X)

	# double centering
	C = np.diag(np.ones(ni)) - (1./ni) * np.ones((ni, ni)) # centering matrix https://en.wikipedia.org/wiki/Centering_matrix
	B = -0.5 * np.matmul(np.matmul(C, D), C)

	# eigen decomposition
	w, v = LA.eig(B)
	ind = np.argsort(-w, axis=0) # sorting descending
	w = w[ind]
	v = v[:, ind]
	
	# create coordinate matrix
	m = 2
	E_m = v[:,0:m] # projection matrix
	lamb_m = np.diag(np.sqrt(w[0:m]))
	X_r = np.matmul(E_m, lamb_m)
	return X_r

def main():
	# load data
	from sklearn import datasets
	X, color = datasets.make_swiss_roll(n_samples=2500)
	# X, color = datasets.make_s_curve(n_samples=2500, random_state=0)

	# timer
	from time import time
	t0 = time()
	# dimred
	X_r = classical_mds(X)
	t1 = time()
	print('[Info] Classical MDS done in {:.2g} sec.'.format(t1 - t0))

	# plot
	fig = plt.figure()

	ax = fig.add_subplot(211, projection='3d')
	ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)

	ax.set_title("Original data")
	ax = fig.add_subplot(212)
	ax.scatter(X_r[:, 0], X_r[:, 1], c=color, cmap=plt.cm.Spectral)
	plt.axis('tight')
	plt.xticks([]), plt.yticks([])
	plt.title('Projected data')
	plt.show()
	input("Press Enter to continue...")
	
	return

if __name__ == '__main__':
	main()
