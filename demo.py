from numpy import *
import pandas as pd

def compute_error_function(c , m ,dataset):
	#Error = 1/N (sum_of_all(y_i - (mx+b))**2)
	#(mx + b) = y
	# N  = no_of_points
	total_error = 0
	for i in range(0,len(dataset)):
		x = dataset[i,0]
		y = dataset[i,1]
		total_error += (y - (m*x+c)) ** 2
	return total_error/float(len(dataset))

def gradient_descent_formula(current_c,current_m,dataset,learning_rate):
	gradient_c = 0
	gradient_m = 0
	N = float(len(dataset))
	for i in range(0,len(dataset)):
		x = dataset[i,0]
		y = dataset[i,1]
		gradient_c += -(2/N) * (y - (current_m * x) + current_c)
		gradient_c += -(2/N) * x * (y - (current_m * x) + current_c)
	new_c = current_c - (gradient_c * learning_rate)
	new_m = current_m - (gradient_m * learning_rate)
	return [new_c,new_m]

def gradient_descent_main(dataset,starting_c,starting_m,learning_rate,iteration):
	c = starting_c
	m = starting_m
	for i in range(iteration):
		c,m = gradient_descent_formula(c,m,array(dataset),learning_rate)
	return [c,m]


def main():
	dataset = genfromtxt('data.csv',delimiter = ',')
	'''
	Hyperparameter - > Tuning knobs
	'''
	learning_rate = 0.0001
	#y=mx + c
	intial_m = 0
	intial_c =0
	# No of times we want to run our iteration
	iteration = 10000
	[c,m] = gradient_descent_main(dataset,intial_c,intial_m,learning_rate,iteration)
	print("intial_c = ",intial_c,"intial_m = ",intial_m,"error = ",compute_error_function(intial_c,intial_m,dataset))
	print("Getting Started /................................................./")
	[c,m] = gradient_descent_main(dataset,intial_c,intial_m,learning_rate,iteration)
	print("c = ",c,"m = ",m, "no of iterations = ",iteration,"learning_rate = ",learning_rate)


if __name__ == '__main__':
	main()