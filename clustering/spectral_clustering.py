"""
Author: Peratham Wiriyathammabhum


"""
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

def spectral_clustering(X, k=3, thres=10**-5, max_iters=200):
	"""
	Perform spectral clustering on an input row matrix X.
	See: http://www.math.ucsd.edu/~fan/research/revised.html
		http://www.math.ucsd.edu/~fan/research/cbms.pdf
	"""
	ni, nd = X.shape
	mu = np.zeros((k, nd))
	krows = np.random.choice(ni, k, replace=False)
	nmu = X[krows]
	nite = 0
	while not terminating_cond(mu, nmu, thres, nite, max_iters):
		mu = nmu

		# e-step: centroid assignments from parameters
		dist = ((X[:,None,:] - mu[None,:,:])**2).sum(axis=2)
		print('[Info] iter: {} SSE: {}'.format(nite, dist.sum()))
		# m-step: estimate latent parameters (centroids)
		cidx = np.argmin(dist, axis=1)
		for i in range(k):
			nmu[i] = X[cidx==i].mean(axis=0)

		nite += 1
		if nite % 10 == 0:
			print('[Info] iter: {}'.format(nite))
	return mu

def terminating_cond(mu, nmu, thres, nite, max_iters):
	if nite >= max_iters:
		print('[Info] terminate at iter: {}'.format(nite))
		return True
	# elif np.linalg.norm(mu - nmu) < thres:
	# 	print('[Info] terminate at iter: {}'.format(nite))
	# 	return True
	else:
		return False

def main(opts):
	k = opts['k']
	max_iters = opts['max_iters']
	n_samples = opts['n_samples']
	from sklearn import datasets
	two_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
	X = two_moons.data
	mu = spectral_clustering(X, k=k, thres=10**-5, max_iters=max_iters)

	# plot
	dist = ((X[:,None,:] - mu[None,:,:])**2).sum(axis=2)
	print('[Info] SSE: {}'.format(dist.sum()))
	cidx = np.argmin(dist, axis=1)
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')
	for c in range(k):
		cluster_members = [X[i] for i in range(len(X)) if cidx[i] == c]    
		cluster_members = np.array(cluster_members)

		ax.scatter(cluster_members[:,0], cluster_members[:,1], cluster_members[:,2], s= 0.5)
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
					  default=3, type=int)
	parser.add_argument('--max_iters', dest='max_iters',
					  help='number of iterations to train',
					  default=200, type=int)
	parser.add_argument('--n_samples', dest='n_samples',
					  help='number of data points',
					  default=2000, type=int)
	args = parser.parse_args()
	opts = vars(args)

	main(opts)
