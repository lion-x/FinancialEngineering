import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as spo


def error_poly(C, data):
    err = np.sum((data[:, 1] - np.polyval(C, data[:, 0])) ** 2)
    return err


def fit_poly(data, error_func, degree=3):
    Cguess = np.poly1d(np.ones(degree + 1, dtype=np.float32))

    x = np.linspace(-10, 10, 21)
    plt.plot(x, np.polyval(Cguess, x), 'm--', linewidth=2.0, label="Initial guess")

    min_result = spo.minimize(error_func, Cguess, args=(data, ), method="SLSQP", options={'disp': True})

    return min_result.x


def fit_line(data, error_func):
    l = np.float32([0, np.mean(data[:, 1])])

    x_ends = np.float32([-5, 5])
    plt.plot(x_ends, l[0] * x_ends + l[1], 'm--', linewidth=2.0, label="Initial guess")

    min_result = spo.minimize(error, l, args=(data, ), method="SLSQP", options={'disp': True})

    return np.poly1d(min_result.x)


def error(line, data):
    err = np.sum((data[:, 1] - (line[0] * data[:, 0] + line[1])) ** 2)
    return err


def f(x):
    y = (x - 1.5)**2 + 0.5
    print("x = {}, y = {}".format(x, y))
    return y


def test_run():
    xguess = 2.0
    min_result = spo.minimize(f, xguess, method='SLSQP', options={'disp': True})
    print("Minima found at: ")
    print("x = {}, y = {}".format(min_result.x, min_result.fun))
    xplot = np.linspace(0.5, 2.5, 21)
    yplot = f(xplot)
    plt.plot(xplot, yplot)
    plt.plot(min_result.x, min_result.fun, 'ro')
    plt.title("Minima of an objective function")
    plt.show()

    l_orig = np.float32([4, 2])
    print("Original line: C0 = {}, C1 = {}".format(l_orig[0], l_orig[1]))
    Xorig = np.linspace(0, 10, 21)
    Yorig = l_orig[0] * Xorig + l_orig[1]
    plt.plot(Xorig, Yorig, 'b--', linewidth=2.0, label="Original line")

    noise_sigma = 3.0
    noise = np.random.normal(0, noise_sigma, Yorig.shape)
    data = np.asarray([Xorig, Yorig + noise]).T
    plt.plot(data[:, 0], data[:, 1], 'go', label="Data points")

    l_fit = fit_line(data, error)

    print(l_fit)

    plt.plot(data[:, 0], l_fit[0] * data[:, 0] + l_fit[1], 'r--', linewidth=2.0, label="Fitted line")
    plt.show()

    l_orig = np.float32([1.625, -10.55, -7.031, 64.63, 51.95])
    print("Original line: C0 = {}, C1 = {}, C2 = {}, C3 = {}, C4 = {}".format(l_orig[0], l_orig[1],
                                                                              l_orig[2], l_orig[3], l_orig[4]))
    Xorig = np.linspace(-10, 10, 21)
    Yorig = np.polyval(l_orig, Xorig)
    print(Xorig)
    print(Yorig)
    plt.plot(Xorig, Yorig, 'b--', linewidth=2.0, label="Original line")

    noise_sigma = 3.0
    noise = np.random.normal(0, noise_sigma, Yorig.shape)
    data = np.asarray([Xorig, Yorig + noise]).T
    plt.plot(data[:, 0], data[:, 1], 'go', label="Data points")

    l_fit = fit_poly(data, error_poly, 4)

    print(l_fit)

    plt.plot(data[:, 0], np.polyval(l_fit, data[:, 0]), 'r--', linewidth=2.0, label="Fitted line")
    plt.show()


if __name__ == '__main__':
    test_run()

