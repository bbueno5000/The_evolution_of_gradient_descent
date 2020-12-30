"""
TODO: docstring
"""
import math
import matplotlib.pyplot as pyplot
import numpy
import random
import sys

class AdaDelta:
    """
    TODO: docstring
    """
    def __call__(self):
        """
        TODO: docstring
        """
        n_steps = 100000
        gradients = [1e-4, 1e-3, 1e-2, 1e-1, 1e2]
        decay_rate = float(sys.argv[1])
        epsilon = float(sys.argv[2])
        for gradient in gradients:
            mean_gradient2 = 0
            mean_step2 = 0
            steps = numpy.zeros(n_steps)
            for i in range(n_steps):
                mean_gradient2 = decay_rate * mean_gradient2 + (1 - decay_rate) * gradient ** 2
                steps[i] = ((mean_step2 + epsilon) / (mean_gradient2 + epsilon)) ** 0.5 * gradient
                mean_step2 = decay_rate * mean_step2 + (1 - decay_rate) * steps[i] ** 2
            pyplot.plot(steps)
            pyplot.ylabel('Absolute step')
            pyplot.xlabel('Iteration number')
        pyplot.legend(gradients, loc='best')
        pyplot.savefig('plot.pdf')
        pyplot.show()

class BatchGradientDescent:
    """
    TODO: docstring
    """
    def __call__(self):
        """
        TODO: docstring
        """
        self.constant()
        self.adaptive()
        self.cricket_chirps()

    def adaptive(self):
        """
        Batch gradient descent with adaptive learning rate.
        """
        x_old = 0
        x_new = 2 # the algorithm starts at x=2
        precision = 0.0001
        x_list, y_list = [x_new], [f(x_new)]
        while abs(x_new - x_old) > precision:
            x_old = x_new
            s_k = -self.f_prime(x_old)
            n_k = scipy.optimize.fmin(self.f2, 0.1, (x_old, s_k), full_output=False, disp=False)
            x_new = x_old + n_k * s_k
            x_list.append(x_new)
            y_list.append(f(x_new))
        print('Local minimum occurs at', float(x_new))
        print('Number of steps:', len(x_list))
        pyplot.figure(figsize=[15, 3])
        pyplot.subplot(1, 3, 1)
        pyplot.scatter(x_list, y_list, c='r')
        pyplot.plot(x_list, y_list, c='r')
        pyplot.plot(x, f(x), c='b')
        pyplot.xlim([-1, 2.5])
        pyplot.title('Gradient descent')
        pyplot.subplot(1, 3, 2)
        pyplot.scatter(x_list, y_list, c='r')
        pyplot.plot(x_list, y_list, c='r')
        pyplot.plot(x, f(x), c='b')
        pyplot.xlim([1.2, 2.1])
        pyplot.ylim([0, 3])
        pyplot.title('zoomed in')
        pyplot.subplot(1, 3, 3)
        pyplot.scatter(x_list, y_list, c='r')
        pyplot.plot(x_list, y_list, c='r')
        pyplot.plot(x, f(x), c='b')
        pyplot.xlim([1.3333, 1.3335])
        pyplot.ylim([0, 3])
        pyplot.title('zoomed in more')
        pyplot.show()

    def constant(self):
        """
        Batch gradient descent with constant learning rate.
        """
        f = lambda x: x**3 - 2 * x**2 + 2
        x = numpy.linspace(-1, 2.5, 1000)
        pyplot.plot(x, f(x))
        pyplot.xlim([-1, 2.5])
        pyplot.ylim([0, 3])
        pyplot.show()
        x_old = 0
        x_new = 2 # the algorithm starts at x=2
        n_k = 0.1 # step size
        precision = 0.0001
        x_list, y_list = [x_new], [f(x_new)]
        while abs(x_new - x_old) > precision:
            x_old = x_new
            s_k = -self.f_prime(x_old)
            x_new = x_old + n_k * s_k
            x_list.append(x_new)
            y_list.append(f(x_new))
        print('Local minimum occurs at:', x_new)
        print('Number of steps:', len(x_list))
        pyplot.figure(figsize=[10, 3])
        pyplot.subplot(1, 2, 1)
        pyplot.scatter(x_list, ny_list, c='r')
        pyplot.plot(x_list, y_list, c='r')
        pyplot.plot(x,f(x), c='b')
        pyplot.xlim([-1, 2.5])
        pyplot.ylim([0, 3])
        pyplot.title('Gradient descent')
        pyplot.subplot(1, 2, 2)
        pyplot.scatter(x_list, y_list, c='r')
        pyplot.plot(x_list, y_list, c='r')
        pyplot.plot(x, f(x), c='b')
        pyplot.xlim([1.2, 2.1])
        pyplot.ylim([0, 3])
        pyplot.title('Gradient descent (zoomed in)')
        pyplot.show()

    def cricket_chirps(self):
        """
        Consider a simple linear regression where we want to see
        how the temperature affects the noises made by crickets.
        """
        data = numpy.loadtxt('SGD_data.txt', delimiter=',')
        pyplot.scatter(data[:, 0], data[:, 1], marker='o', c='b')
        pyplot.title('cricket chirps vs temperature')
        pyplot.xlabel('chirps/sec for striped ground crickets')
        pyplot.ylabel('temperature in degrees Fahrenheit')
        pyplot.xlim([13,21])
        pyplot.ylim([65,95])
        pyplot.show()
        h = lambda theta_0,theta_1,x: theta_0 + theta_1*x
        x = data[:, 0]
        y = data[:, 1]
        m = len(x)
        theta_old = numpy.array([0.0, 0.0])
        theta_new = numpy.array([1.0, 1.0]) # yhe algorithm starts at [1,1]
        n_k = 0.001 # step size
        precision = 0.001
        num_steps = 0
        s_k = float('inf')
        while numpy.linalg.norm(s_k) > precision:
            num_steps += 1
            theta_old = theta_new
            s_k = -self.grad_J(x, y, m, theta_old[0], theta_old[1])
            theta_new = theta_old + n_k * s_k
        print('Local minimum occurs where:')
        print('theta_0 =', theta_new[0])
        print('theta_1 =', theta_new[1])
        print('This took', num_steps, 'steps to converge')
        actualvalues = scipy.stats.linregress(x, y)
        print('Actual values for theta are:')
        print('theta_0 =', actualvalues.intercept)
        print('theta_1 =', actualvalues.slope)
        xx = numpy.linspace(0,21,1000)
        pyplot.scatter(data[:, 0], data[:, 1], marker='o', c='b')
        pyplot.plot(xx,h(theta_new[0], theta_new[1], xx))
        pyplot.xlim([13, 21])
        pyplot.ylim([65, 95])
        pyplot.title('cricket chirps vs temperature')
        pyplot.xlabel('chirps/sec for striped ground crickets')
        pyplot.ylabel('temperature in degrees Fahrenheit')
        pyplot.show()

    def decrease(self):
        """
        Batch gradient descent with decreasing-constant learning rate.
        """
        x_old = 0
        x_new = 2 # the algorithm starts at x=2
        n_k = 0.17 # step size
        precision = 0.0001
        t, d = 0, 1
        x_list, y_list = [x_new], [f(x_new)]
        while abs(x_new - x_old) > precision:
            x_old = x_new
            s_k = -self.f_prime(x_old)
            x_new = x_old + n_k * s_k
            x_list.append(x_new)
            y_list.append(f(x_new))
            n_k = n_k / (1 + t * d)
            t += 1
        print('Local minimum occurs at:', x_new)
        print('Number of steps:', len(x_list))

    def f_prime(self, x):
        """
        returns the value of the derivative of our function
        """
        return 3 * x**2 - 4 * x

    def f2(self, n, x, s):
        """
        We setup this function to pass into the fmin algorithm.
        """
        x = x + n*s
        return f(x)

    def grad_J(self, x, y, m, theta_0, theta_1):
        """

        """
        returnValue = numpy.array([0.0, 0.0])
        for i in range(m):
            returnValue[0] += (h(theta_0, theta_1, x[i]) - y[i])
            returnValue[1] += (h(theta_0, theta_1, x[i]) - y[i]) * x[i]
        returnValue = returnValue / (m)
        return returnValue

    def J(self, x, y, m, theta_0, theta_1):
        """

        """
        returnValue = 0
        for i in range(m):
            returnValue += (h(theta_0, theta_1, x[i]) - y[i])**2
        returnValue = returnValue / (2 * m)
        return returnValue

class Nesterov:
    """
    TODO: docstring
    """
    def calc_numerical_gradient(self, func, x, delta_x):
        """
        Function for computing gradient numerically.
        """
        val_at_x = func(x)
        val_at_next = func(x + delta_x)
        return (val_at_next - val_at_x) / delta_x

    def nesterov_descent(
        self,
        func,
        L,
        dimension,
        init_x=None,
        numerical_gradient=True,
        delta_x=0.0005,
        gradient_func=None,
        epsilon=None):
        """
        TODO: docstring
        """
        assert delta_x > 0, "Step must be positive."
        if init_x is None:
            x = numpy.zeros(dimension)
        else:
            x = init_x
        if epsilon is None:
            epsilon = 0.05
        lambda_prev = 0
        lambda_curr = 1
        gamma = 1
        y_prev = x
        alpha = 0.05 / (2 * L)
        if numerical_gradient:
            gradient = calc_numerical_gradient(func, x, delta_x)
        else:
            gradient = gradient_func(x)
        while numpy.linalg.norm(gradient) >= epsilon:
            y_curr = x - alpha * gradient
            x = (1 - gamma) * y_curr + gamma * y_prev
            y_prev = y_curr
            lambda_tmp = lambda_curr
            lambda_curr = (1 + math.sqrt(1 + 4 * lambda_prev * lambda_prev)) / 2
            lambda_prev = lambda_tmp
            gamma = (1 - lambda_prev) / lambda_curr
            if numerical_gradient:
                gradient = calc_numerical_gradient(func, x, delta_x)
            else:
                gradient = gradient_func(x)
        return x

class StochasticGradientDescent:
    """
    TODO: docstring
    """
    def __call__(self):
        """
        TODO: docstring
        """
        f = lambda x: x * 2 + 17 + numpy.random.randn(len(x)) * 10
        x = numpy.random.random(500000) * 100
        y = f(x) 
        m = len(y)
        x_shuf, y_shuf = list(), list()
        index_shuf = range(len(x))
        random.shuffle(index_shuf)
        for i in index_shuf:
            x_shuf.append(x[i])
            y_shuf.append(y[i])
        h = lambda theta_0, theta_1, x: theta_0 + theta_1  *x
        cost = lambda theta_0, theta_1, x_i, y_i: 0.5 * (h(theta_0, theta_1, x_i) - y_i)**2
        theta_old = numpy.array([0.0, 0.0])
        theta_new = numpy.array([1.0, 1.0]) # the algorithm starts at [1, 1]
        n_k = 0.000005 # step size
        iter_num = 0
        s_k = numpy.array([float('inf'), float('inf')])
        sum_cost = 0
        cost_list = list()
        for j in range(10):
            for i in range(m):
                iter_num += 1
                theta_old = theta_new
                s_k[0] = (h(theta_old[0], theta_old[1], x[i]) - y[i])
                s_k[1] = (h(theta_old[0], theta_old[1], x[i]) - y[i]) * x[i]
                s_k = (-1) * s_k
                theta_new = theta_old + n_k * s_k
                sum_cost += cost(theta_old[0], theta_old[1], x[i], y[i])
                if i + 1 % 10000 == 0:
                    cost_list.append(sum_cost/10000.0)
                    sum_cost = 0   
        print('Local minimum occurs where:')
        print('theta_0 =', theta_new[0])
        print('theta_1 =', theta_new[1])
        iterations = numpy.arange(len(cost_list)) * 10000
        plt.plot(iterations, cost_list)
        plt.xlabel('iterations')
        plt.ylabel('avg cost')
        plt.show()

def main(argv):
    """
    TODO: docstring
    """
    batch_gradient_descent = BatchGradientDescent()
    batch_gradient_descent()
    stochastic_gradient_descent = StochasticGradientDescent()
    stochastic_gradient_descent()

if __name__ == '__main__':
    main(sys.argv)
