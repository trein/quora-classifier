import math as math
import numpy as np
import scipy.integrate as integrate

def univariate_integrate(f, n=1000, start=0, end=2):
    return sum([f(i)*(abs(start-end)/float(n)) for i in np.arange(start, end, abs(start-end)/float(n))])

def double_integrate(f, x_0, x_n, y_0, y_n):
    return integrate.dblquad(f, x_0, x_n, lambda x: y_0, lambda x: y_n, epsabs=1.49e-02, epsrel=1.49e-02)[0]

class PJoin:

    def __init__(self, x_array, y_array):
        self.x_array = x_array
        self.y_array = y_array

    def __call__(self, x, y):
        kernel_part = 0.0
        for x_i, y_i in zip(self.x_array, self.y_array):
            kernel_part += self.kernel(x, x_i, self.x_array) * self.kernel(y, y_i, self.y_array)

        return kernel_part * 1.0/len(self.x_array)

    def kernel(self, var, var_i, var_array):
        d = len(var_array)
        z = var - var_i
        sigma = np.cov(z)
        h = self.compute_h(var_array)

        num = - z**2 * 1.0/sigma
        den = 2 * (h**2)

        return np.exp(num / den) / (2 * math.pi) ** (1.0*d/2) * (h**d) * (np.abs(sigma) ** (1.0/2))

    def compute_h(self, var_array):
        number_examples = len(var_array)
        return 1.06 * np.std(var_array, axis=0) * (number_examples ** (-1.0/5))

if __name__ == "__main__":
    function = lambda x : x
    print "Univariate Integral", univariate_integrate(function)

    def f(x, y):
        return x + y
    print "Double Integral", double_integrate(f, -1, 1, -1, 1)


    x = np.array([1,2,3])
    y = np.array([1,1,0])
    print "Double Integral", double_integrate(PJoin(x, y), x.min(), x.max(), y.min(), y.max())