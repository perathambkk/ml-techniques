"""
From matplotlib (mpl) tutorial,
Link: https://matplotlib.org/3.1.1/gallery/specialty_plots/hinton_demo.html

"""
import numpy as np
import matplotlib.pyplot as plt

def hinton(matrix, max_weight=None, ax=None):
	"""Draw Hinton diagram for visualizing a weight matrix."""
	ax = ax if ax is not None else plt.gca()

	if not max_weight:
		max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

	ax.patch.set_facecolor('gray')
	ax.set_aspect('equal', 'box')
	ax.xaxis.set_major_locator(plt.NullLocator())
	ax.yaxis.set_major_locator(plt.NullLocator())

	for (x, y), w in np.ndenumerate(matrix):
		color = 'white' if w > 0 else 'black'
		size = np.sqrt(np.abs(w) / max_weight)
		if size > 0:
			rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
								 facecolor=color, edgecolor=color)
		else:
			size = 1
			rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
								 facecolor='gray', edgecolor='gray')
		ax.add_patch(rect)

	ax.autoscale_view()
	ax.invert_yaxis()
	return


if __name__ == '__main__':
	# Fixing random state for reproducibility
	# np.random.seed(19680801)
	# np.random.seed(19680801)

	hinton(np.random.rand(2, 2) - 0.5)
	plt.show()
	input("Press Enter to continue...")
