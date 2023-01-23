# Numeric Integration

from math import exp

def integ1(f, a, b, n=100):
	# Trapezoidal
	w = (b - a) / n
	seq = (a + i * w for i in range(n + 1))
	s = 2 * sum(map(f, seq))
	s = s - f(a) - f(b)
	return s * w / 2

def integ2(f, a, b, n=100):
	# Mid-point
	w = (b - a) / n
	seq = (a + (i + .5) * w for i in range(n))
	s = sum(map(f, seq))
	return s * w

def integ3(f, a, b, n=100):
	# Simpson's Rule (Thirds) or (1+4+1 / 6)
	w = (b - a) / n
	s = 0
	for i in range(n):
		x1 = a + i * w
		x3 = x1 + w
		x2 = (x1 + x3) / 2
		s += f(x1) + 4 * f(x2) + f(x3)
	return s * w / 6

def integ4(f, a, b, n=100):
	# Simpson's Second Rule (Three Eighths) or (1+3+3+1 / 8)
	w = (b - a) / n
	s = 0
	for i in range(n):
		x1 = a + i * w
		x4 = x1 + w
		x2 = (2 * x1 + x4) / 3
		x3 = (x1 + 2 * x4) / 3
		s += f(x1) + 3 * f(x2) + 3 * f(x3) + f(x4)
	return s * w / 8

def square(x):
	return x * x

def func2(t):
	return 3 * t ** 2 * exp(t ** 3)

a = 0      #3  # lower limit
b = 1      #5  # upper limit
f = func2      #square

for n in range(1, 201):
	print('{:>4}   {:<22.18}   {:<22.18}   {:<22.18}   {:<22.18}'.format(
			n, 
			integ1(f, a, b, n), 
			integ2(f, a, b, n), 
			integ3(f, a, b, n), 
			integ4(f, a, b, n)))

print('--------')
print('       {:<22.18}   {:<22.18}   {:<22.18}   {:<22.18}'.format(
		integ1(f, a, b), 
		integ2(f, a, b), 
		integ3(f, a, b), 
		integ4(f, a, b)))
