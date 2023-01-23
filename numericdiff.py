DELTA_X = h = 1e-6

def square(x):
	return x * x

def cube(x):
	return x * x * x

def diff_f(f, x):
	return (f(x + h) - f(x)) / h

def diff_b(f, x):
	return (f(x) - f(x - h)) / h

def diff_c(f, x):
	return (f(x + h) - f(x - h)) / 2 / h

def stencil_5(f, x):
	return (-f(x + 2 * h) + 8 * f(x + h) - 8 * f(x - h) + f(x - 2 * h)) / 12 / h

gradients =[diff_f, diff_b, diff_c, stencil_5]
funcs = [square, cube]
x_values = [4, 3]

print('gradient\tfunction\tx_value\tf(x)\tgradient')
for f, x in zip(funcs, x_values):
	print()
	for g in gradients:
		print(f'{g.__name__}\t{f.__name__}\t{x}\t{f(x)}\t{g(f, x)}')

