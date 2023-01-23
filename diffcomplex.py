
h = 1e-6

def square(x):
    return x * x

def var(x):
    return complex(x, h)

def diff_cx(f, x):
    return f(x).imag / h

print('func\tx\tz\t\tfunc(x)\t\tdiff(func, z)')

f = square
x = 4
z = var(x)
d = diff_cx

print(f'{f.__name__}\t{x}\t{z}\t{f(x)}\t\t{d(f, z)}')

