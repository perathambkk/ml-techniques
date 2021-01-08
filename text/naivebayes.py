"""
Author: Peratham Wiriyathammabhum


"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import scipy.sparse.linalg as linalg

def naivebayes(X):
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

def tfidf():

	return

def main(opts):
	k = opts['k']

	# load data
	categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
	from sklearn.datasets import fetch_20newsgroups
	from sklearn.feature_extraction.text import CountVectorizer
	count_vect = CountVectorizer()
	X_train_counts = count_vect.fit_transform(twenty_train.data)

	# tf-idf

	# clustering
	_, cidx = spectral_clustering(X, mode=mode, k=k, knn=knn, eta=eta, sigma=sigma, max_iters=max_iters)

	# plot
	
	return

if __name__ == '__main__':
	import argparse

	parser = argparse.ArgumentParser(description='run naivebayes.')
	parser.add_argument('--k', dest='k',
					  help='number of clusters',
					  default=2, type=int)
	args = parser.parse_args()
	opts = vars(args)

	main(opts)
