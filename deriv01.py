
from math import sin, cos, pi

def derivative(func, x, h):
    return (func(x + h) - func(x)) / h

x = pi / 4
diffs = {}

for j in range(21):
    h = 10 ** -j
    deriv = derivative(sin, x, h)
    exact = cos(x)
    diff = abs(deriv - exact)
    diffs[j] = diff

print(diffs)
for k, v in diffs.items():
    print(f'{k}:\t{v}')

