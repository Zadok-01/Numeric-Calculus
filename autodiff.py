# autodiff.py  --  Automatic Differentiation

import math

# New Number Class  ############################
class DN:
    '''Instances of this class behave like a floats for input to 
    functions, but any operations simultaneously calculate its derivative.
    (DN is an abbreviation of Diff-Number)
    
    Creating a DN
    -------------
    New numbers are created using:  DN(x, dx)
    Make constants (not varying with respect to x) using:  DN(3.5)
    Make variables (ie x values) with:  DN(3.5, 1)
    The shortcut for DN(3.5, 1) is:  v(3.5)
    
    Accessing DNs
    -------------
    Given an instance of a DN called a:
    To obtain the 'real number' portion use a.n or the alternative n(a).
    This alternative can also be used on floats or ints and will return 
    their value in case that the class of the argument is unknown.
    To obtain the derivative portion use a.d or the alternative d(a).
    This alternative can also be used with floats or ints, but will 
    return the value 0.
    
    Functions of One Variable
    -------------------------
    >>> f = lambda x: 3 * x ** 3 * sin(4 * x) ** 2
    >>> y = f(v(1.2))      # Evaluate y at x = 1.2
    >>> n(y)               # f(1.2)
    5.144310922218376
    >>> d(y)               # f'(1.2)
    9.245937170106238
    
    Multivariate Functions
    ----------------------
    This class can also be used to compute gradients of multivariate 
    functions by making one of the arguments variable and the keeping 
    the others constant:
    
    >>> f = lambda x, y:  x * y + sin(x)
    >>> z = f(2.5, 3.5)        # Evaluate at (2.5, 3.5)
    >>> n(z)
    9.348472144103956
    >>> d(f(v(2.5), 3.5))      # Partial diff with respect to x
    2.6988563844530664
    >>> d(f(2.5, v(3.5)))      # Partial diff with respect to y
    2.5
    >>> gradient(f, (2.5, 3.5))
    (2.6988563844530664, 2.5)
    '''
    
    def __init__(self, n, d=0):
        if isinstance(n, DN):
            self.n, self.d = n.n, n.d
        else:
            self.n, self.d = n, d
    
    def __repr__(self):
        return f'DN({self.n}, {self.d})'
    
    def __abs__(a):
        return abs(n(a))
    
    def __add__(a, b):
        return DN(n(a) + n(b), d(a) + d(b))
    
    def __sub__(a, b):
        return DN(n(a) - n(b), d(a) - d(b))
    
    def __mul__(a, b):
        return DN(n(a) * n(b), n(a) * d(b) + n(b) * d(a))
    
    def __truediv__(a, b):
        return DN(n(a) / n(b), (n(b) * d(a) - n(a) * d(b)) / n(b) ** 2)
    
    def __pow__(a, b):
        return DN(n(a) ** n(b),
                (n(b) * n(a) ** (n(b) - 1) * d(a) if d(a) else 0) + 
                (math.log(n(a)) * n(a) ** n(b) * d(b) if d(b) else 0))
    
    def __floordiv__(a, b):
        return DN(n(a) // n(b), 0)
    
    def __mod__(a, b):
        return DN(n(a) % n(b), d(a) - n(a) // n(b) * d(b))
    
    def __pos__(a):
        return a

    def __neg__(a):
        return DN(-n(a), -d(a))
    
    __radd__ = __add__
    __rmul__ = __mul__
    
    def __rsub__(a, b):
        return DN(b) - a
    
    def __rtruediv__(a, b):
        return DN(b) / a
    
    def __rpow__(a, b):
        return DN(b) ** a

    def __rmod__(a, b):
        return DN(b) % a

    def __rfloordiv__(a, b):
        return DN(b) // a
    
    # Add other operations as required.
    # Missing are in-place operations such as += etc.

# Shortcuts ####################################
v = lambda x: DN(x, 1)
n = lambda x: getattr(x, 'n', x)
d = lambda x: getattr(x, 'd', 0)

# Functions and Constants ######################
sqrt = lambda a: DN(math.sqrt(n(a)), d(a) / (2 * math.sqrt(n(a))))
log = lambda a: DN(math.log(n(a)), d(a) / n(a))
log2 = lambda a: DN(math.log2(n(a)), d(a) / (n(a) * math.log(2)))
log10 = lambda a: DN(math.log10(n(a)), d(a) / (n(a) * math.log(10)))
log1p = lambda a: DN(math.log1p(n(a)), d(a) / (n(a) + 1))
exp = lambda a: DN(math.exp(n(a)), math.exp(n(a)) * d(a))
expm1 = lambda u: DN(math.expm1(n(a)), math.exp(n(a)) * d(a))
sin = lambda a: DN(math.sin(n(a)), math.cos(n(a)) * d(a))
cos = lambda a: DN(math.cos(n(a)), -math.sin(n(a)) * d(a))
tan = lambda a: DN(math.tan(n(a)), d(a) / math.cos(n(a)) ** 2)
sinh = lambda a: DN(math.sinh(n(a)), math.cosh(n(a)) * d(a))
cosh = lambda a: DN(math.cosh(n(a)), math.sinh(n(a)) * d(a))
tanh = lambda a: DN(math.tanh(n(a)), d(a) / math.cosh(n(a)) ** 2)
asin = lambda a: Num(math.asin(n(a)), d(a) / math.sqrt(1 - n(a) ** 2))
acos = lambda a: DN(math.acos(n(a)), -d(a) / math.sqrt(1 - n(a) ** 2))
atan = lambda a: DN(math.atan(n(a)), d(a) / (1 + n(a) ** 2))
asinh = lambda a: DN(math.asinh(n(a)), d(a) / math.hypot(n(a), 1))
acosh = lambda a: DN(math.acosh(n(a)), d(a) / math.sqrt(n(a) ** 2 - 1))
atanh = lambda a: DN(math.atanh(n(a)), d(a) / (1 - n(a) ** 2))
radians = lambda a: DN(math.radians(n(a)), math.radians(d(a)))
degrees = lambda a: DN(math.degrees(n(a)), math.degrees(d(a)))
erf = lambda a: DN(math.erf(n(a)), 
        2 / math.sqrt(math.pi) * math.exp(-(n(a) ** 2)) * d(a))
erfc = lambda a: DN(math.erfc(n(a)), 
        -2 / math.sqrt(math.pi) * math.exp(-(n(a) ** 2)) * d(a))
hypot = lambda a, b: DN(math.hypot(n(a), n(b)), 
        (n(a) * d(a) + n(b) * d(b)) / math.hypot(n(a), n(b)))
fsum = lambda a: DN(math.fsum(map(n, a)), math.fsum(map(d, a)))
fabs = lambda a: abs(DN(a))
fmod = lambda a, b: DN(a) % b
copysign = lambda a, b: DN(math.copysign(n(a), n(b)), 
        d(a) if math.copysign(1, n(a) * n(b)) > 0  else -d(a))
ceil = lambda a: DN(math.ceil(n(a)), 0)
floor = lambda a: DN(math.floor(n(a)), 0)
trunc = lambda a: DN(math.trunc(n(a)), 0)
pi = DN(math.pi)
e = DN(math.e)

    # Add other functions and constants as required.
    # They need not be included in the math library.

# Advanced Techniques ##########################
def partial_d(func, point, index):
    ''' Partial derivative at a given point
    
    >>> func = lambda x, y:  x * y + sin(x)
    >>> point = (2.5, 3.5)
    >>> partial_d(func, point, 0)             # Partial with respect to x
    2.6988563844530664
    >>> partial_d(func, point, 1)             # Partial with respect to y
    2.5
    '''
    return d(func(*[DN(x, i==index) for i, x in enumerate(point)]))

def gradient(func, point):
    '''Vector of the partial derivatives of a scalar field
    
    >>> func = lambda x, y:  x * y + sin(x)
    >>> point = (2.5, 3.5)
    >>> gradient(func, point)
    (2.6988563844530664, 2.5)
    '''
    return tuple(partial_d(func, point, index) for index in range(len(point)))

def directional_derivative(func, point, direction):
    ''' The dot product of the gradient and a direction vector.
    Computed directly with a single function call.
    
    >>> func = lambda x, y:  x * y + sin(x)
    >>> point = (2.5, 3.5)
    >>> direction = (1.5, -2.2)
    >>> directional_derivative(func, point, direction)
    -1.4517154233204006
    
    Same result as separately computing and dotting the gradient:
    >>> math.fsum(g * d for g, d in zip(gradient(func, point), direction))
    -1.4517154233204002
    '''
    return d(func(*map(DN, point, direction)))

def divergence(F, point):
    ''' Sum of the partial derivatives of a vector field
    
    Example 1:
    >>> F = lambda x, y, z: (x*y+sin(x)+3*x, x-y-5*x, cos(2*x)-sin(y)**2)
    >>> divergence(F, (3.5, 2.1, -3.3))
    3.163543312709203
    
    Using result from Wolfram Alpha calculator:
    # https://www.wolframalpha.com/input/?i=div%28x*y%2Bsin%28x%29%2B3*x%2C+x-y-5*x%2C+cos%282*x%29-sin%28y%29**2%29
    >>> x, y, z = (3.5, 2.1, -3.3)
    >>> math.cos(x) + y + 2      # from Wolfram Alpha
    3.1635433127092036
    
    Example 2:
    >>> F = lambda x, y, z: (8 * exp(-x), cosh(z), - y**2)
    >>> divergence(F, (2, -1, 4))
    -1.0826822658929016
    
    Using result from YouTube Tutorial:
    # https://www.youtube.com/watch?v=S2rT2zK2bdo
    >>> x, y, z = (2, -1, 4)
    >>> -8 * math.exp(-x)      # from YouTube tutorial
    -1.0826822658929016
    '''
    return math.fsum(d(F(*[DN(x, i==index) for i, x in enumerate(point)])[index]) for index in range(len(point)))

def curl(F, point):
    ''' Rotation around a vector field
    
    Example 1:
    >>> F = lambda x, y, z: (x*y+sin(x)+3*x, x-y-5*x, cos(2*x)-sin(y)**2)
    >>> curl(F, (3.5, 2.1, -3.3))
    (0.8715757724135882, 1.3139731974375781, -7.5)
    
    Using result from Wolfram Alpha calculator:
    # http://www.wolframalpha.com/input/?i=curl+%7Bx*y%2Bsin(x)%2B3*x,+x-y-5*x,+cos(2*x)-sin(y)%5E2%7D
    >>> x, y, z = (3.5, 2.1, -3.3)
    >>> (-2 * math.sin(y) * math.cos(y), 2 * math.sin(2 * x), -x - 4)      # from Wolfram Alpha
    (0.8715757724135882, 1.3139731974375781, -7.5)
    
    Example 2:
    >>> F = lambda x, y, z: (8 * exp(-x), cosh(z), - y**2)
    >>> curl(F, (2, -1, 4))
    (-25.289917197127753, 0.0, 0.0)
    
    Using result from YouTube Tutorial:
    # https://www.youtube.com/watch?v=S2rT2zK2bdo
    >>> x, y, z = (2, -1, 4)
    >>> (-(x * y + math.sinh(z)), 0.0, 0.0)      # from YouTube tutorial
    (-25.289917197127753, 0.0, 0.0)
    '''
    x, y, z = point
    _, Fyx, Fzx = map(d, F(v(x), y, z))
    Fxy, _, Fzy = map(d, F(x, v(y), z))
    Fxz, Fyz, _ = map(d, F(x, y, v(z)))
    return (Fzy - Fyz, Fxz - Fzx, Fyx - Fxy)

# Tests ########################################
def examples_basic():
    
    def square(x):
        return x ** 2
    
    def my_func1(x):
        return x ** 2 * cos(x / 2)
    
    def my_func2(x):
        return ((5 * x ** 3) - sin(2 * x) ** 2) / sqrt(x - 1)
    
    funcs = [square, my_func1, my_func2]
    x_values = [3, 2, 1.5]
    for f, x in zip(funcs, x_values):
        x = v(x)
        y = f(x)
        gradient = d(y)
        print(f'Value of {f.__name__} at {n(x)} is {n(y)}')
        print(f'Gradient of {f.__name__} at {n(x)} is {gradient}')
        print()
    # Expected results: (9, 6), (2.161209, 0.478267), (23.836690, 24.683324)

def examples_advanced():
    # River flow example inspired by:
    # https://www.youtube.com/watch?v=vvzTEbp9lrc
    print('River Flow')
    W = 20      # width of river in meters
    C = 0.1     # max flow divided by (W/2)**2
    F = lambda x, y, z:  (0.0, C * x * (W - x), 0.0)
    for x in range(W+1):
        print(f'{x} : {curl(F, (x, 0, 0))}')

if __name__ == '__main__':
    examples_basic()
    print('----------')
    examples_advanced()

