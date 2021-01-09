"""
Author: Peratham Wiriyathammabhum


"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
# import scipy.sparse.linalg as linalg
from numpy import linalg as LA
from numpy import inf

from graph_utils import geodesic_graph

def isomap(X, mode='affinity', knn=3, eta=0.01):
	"""
	Perform isomap on an input row matrix X.
	See: Tenenbaum, Joshua B., Vin De Silva, and John C. Langford. 
	"A global geometric framework for nonlinear dimensionality reduction." 
	science 290.5500 (2000): 2319-2323. http://www.robots.ox.ac.uk/~az/lectures/ml/tenenbaum-isomap-Science2000.pdf 
	"""
	ni, nd = X.shape

	G = geodesic_graph(X, mode=mode, knn=knn, eta=eta)
	# The operator t is defined by t(D) = -HSH/2, where S
	# is the matrix of squared distances {Sij = Dij**2}, and H is
	# the “centering matrix” {Hij = dij - 1/N} 
	G[G == inf] = 0
	G[np.isnan(G)] = 0
	
	G = G ** 2
	G = G - sum(G, 1).T / ni
	G = G - sum(G, 1) / ni
	G = G + sum(G[:]) / (ni ** 2)
	G = -0.5 * G

	G[G == inf] = 0
	G[np.isnan(G)] = 0

	w, v = LA.eig(G)
	ind = np.argsort(-w, axis=0) # sorting descending
	w = w[ind]
	v = v[:, ind]
	sw = np.sqrt(w)

	v = v * sw

	X_r = v[:,0:2] # projection matrix
	return X_r

def main(opts):
	knn = opts['knn']
	eta = opts['eta']
	mode = opts['mode']

	# load data
	from sklearn import datasets
	X, color = datasets.make_swiss_roll(n_samples=2500)

	# dimred
	X_r = isomap(X, mode=mode, knn=knn, eta=eta)

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
	import argparse

	parser = argparse.ArgumentParser(description='run isomap.')
	parser.add_argument('--knn', dest='knn',
					  help='number of neighbors',
					  default=4, type=int)
	parser.add_argument('--eta', dest='eta',
					  help='distance eta',
					  default=0.01, type=int)
	parser.add_argument('--mode', dest='mode',
					  help='graph construction mode \in {affinity, nearestneighbor}',
					  default='affinity', type=str)
	args = parser.parse_args()
	opts = vars(args)

	main(opts)
