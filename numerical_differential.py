def derivative(f, x, h):
    return (f(x + h) - f(x)) / h

def func(x):
    return 3 * (x ** 2) + 5

def ans(x):
    return 6 * x

h = 0.0000001

# compute numerical differential
start, end, step = 5, 6 ,.1

i = start
while i < end:
    d = derivative(func, i, h)
    a = ans(i)
    print(f'{d}\t{a}\t{d - a}')
    i += step

