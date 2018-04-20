import numpy as np 
import pandas as pd 
from sklearn.datasets import load_iris
X = list(np.arange(1, 21, step=1))
Y = list(np.arange(1, 21, step=1))
# y = mx + b
def linear_reg(X, Y):
	slope = (Y[-1] - Y[0]) / (X[-1] - X[0])
	randomind = np.random.randint(len(X))
	plugin_tuple = X[randomind], Y[randomind]
	b = plugin_tuple[1] - slope + plugin_tuple[0] # y - mx = b
	output_x = list(np.arange(X[-1], X[-1]*2, step=1))
	output_y = list(np.add(np.multiply(slope, output_x), b))
	return output_y
# print(linear_reg(X, Y))


# regression line: b0 + b1(x1) + b2(x2) + b3(x3) = 3 weights for each x, plus a constant. weights will be randomly init
# logistic equation: 1 / 1 + e^(-regression line)
mapper = lambda x: 1 if x > 0.5 else 0

def gradient_descent(x, y, lr, iters):
	N = len(x)
	weights = np.ones(x.shape[1])
	for i in range(iters): 
		# logistic function
		h = np.dot(x, weights)
		loss = h - y
		grad = np.dot(x.T, loss) / N
		weights -= np.multiply(lr, grad)
	return weights

def log_reg(train_x, train_y, test_x, lr, iters):
	weights = list(gradient_descent(train_x, train_y, lr, iters))
	constant = np.random.rand()
	feat_1 = np.array([i[0] for i in test_x])
	feat_2 = np.array([i[1] for i in test_x])
	feat_3 = np.array([i[2] for i in test_x])
	feat_4 = np.array([i[3] for i in test_x])
	h = weights[0]*feat_1 + weights[1]*feat_2 + weights[2]*feat_3 + weights[3]*feat_4 
	pred_y = 1 / (1 + np.exp(-h))
	pred_y = list(map(mapper, pred_y))
	return pred_y

if __name__ == '__main__':
	
	iris = load_iris()

	# make binary classification 
	toremove = len(iris.target[iris.target==2])
	X = iris.data[:-toremove]
	y = iris.target[iris.target!=2] 
	
	from sklearn.cross_validation import train_test_split
	train_x, test_x, train_y, test_y = train_test_split(X, y)
	
	lr = 0.0005
	iters = 100000
	pred_y = log_reg(train_x, train_y, test_x, lr, iters)

	count = 0
	for ind, x in enumerate(pred_y):
		if test_y[ind] != x:
			count += 1

	print('Accuracy:', 1 - count/len(pred_y))