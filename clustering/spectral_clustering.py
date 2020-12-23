"""
Author: Peratham Wiriyathammabhum


"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import scipy.sparse.linalg as linalg
from graph_utils import *
from kmeans import kmeans

def spectral_clustering(X, mode='affinity', k=3, knn=4, eta=0.01, sigma=2.5, max_iters=200):
	"""
	Perform spectral clustering on an input row matrix X.
	mode \in {'affinity','neighborhood','gaussian'}
	See: http://www.math.ucsd.edu/~fan/research/revised.html
		http://www.math.ucsd.edu/~fan/research/cbms.pdf
	"""
	ni, nd = X.shape
	L = laplacian_graph(X, mode='affinity', knn=knn, eta=eta, sigma=sigma)

	vals, vecs = linalg.eigs(L, k=k, which='SR')
	# ind = np.argsort(vals, axis=0)
	# vals = vals[ind]
	# vecs = vecs[:, ind]

	mu = kmeans(vecs, k=k, thres=10**-5, max_iters=max_iters)
	
	dist = ((vecs[:,None,:] - mu[None,:,:])**2).sum(axis=2)
	cidx = np.argmin(dist, axis=1)
	return mu, cidx

def main(opts):
	k = opts['k']
	knn = opts['knn']
	eta = opts['eta']
	sigma = opts['sigma']
	n_samples = opts['n_samples']
	mode = opts['mode']
	max_iters = opts['max_iters']

	# load data
	from sklearn import datasets
	two_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
	X, y = two_moons
	# data plot
	fig = plt.figure()
	ax = fig.add_subplot()

	for c in range(2):
		cluster_members = [X[i] for i in range(len(X)) if y[i] == c]    
		cluster_members = np.array(cluster_members)
		ax.scatter(cluster_members[:,0], cluster_members[:,1],  s= 0.5)
	input("Press Enter to continue...")

	# clustering
	_, cidx = spectral_clustering(X, mode=mode, k=k, knn=knn, eta=eta, sigma=sigma, max_iters=max_iters)

	# plot
	fig = plt.figure()
	ax = fig.add_subplot()
	for c in range(2):
		cluster_members = [X[i] for i in range(len(X)) if cidx[i] == c]    
		cluster_members = np.array(cluster_members)

		ax.scatter(cluster_members[:,0], cluster_members[:,1],  s= 0.5)
	input("Press Enter to continue...")
	"""
	For details see: 
	http://www.cs.cmu.edu/~aarti/Class/10701/readings/Luxburg06_TR.pdf
	https://ai.stanford.edu/~ang/papers/nips01-spectral.pdf
	https://www.cs.upc.edu/~argimiro/mytalks/jmda.pdf
	"""
	return

if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser(description='run k-means.')
	parser.add_argument('--k', dest='k',
					  help='number of clusters',
					  default=2, type=int)
	parser.add_argument('--knn', dest='knn',
					  help='number of neighbors',
					  default=4, type=int)
	parser.add_argument('--eta', dest='eta',
					  help='distance eta',
					  default=0.01, type=int)
	parser.add_argument('--sigma', dest='sigma',
					  help='kernel width',
					  default=2.5, type=int)
	parser.add_argument('--n_samples', dest='n_samples',
					  help='number of data points',
					  default=2000, type=int)
	parser.add_argument('--mode', dest='mode',
					  help='graph construction mode \in {affinity, nearestneighbor, gaussian}',
					  default='affinity', type=str)
	parser.add_argument('--max_iters', dest='max_iters',
					  help='number of kmeans iterations to train',
					  default=200, type=int)
	args = parser.parse_args()
	opts = vars(args)

	main(opts)
