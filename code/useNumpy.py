import numpy as np
import time


def get_max_index(a):
    return np.argmax(a)


def manual_mean(arr):
    sum = 0
    for i in range(0, arr.shape[0]):
        for j in range(0, arr.shape[1]):
            sum = sum + arr[i, j]
    return sum / arr.size


def numpy_mean(arr):
    return arr.mean()


def how_long(func, *args):
    t0 = time.time()
    result = func(*args)
    t1 = time.time()
    return result, t1 - t0


def test_run():
    print(np.array([(2, 3, 4), (5, 6, 7)]))
    print(np.empty(5))
    print(np.empty((5, 4, 3)))
    print(np.ones((5, 4), dtype=np.int_))
    print(np.random.random((5, 4)))
    print(np.random.rand(5, 4))
    print(np.random.normal(size=(2, 3)))
    print(np.random.normal(50, 10, size=(2, 3)))
    print(np.random.randint(10))
    print(np.random.randint(0, 10))
    print(np.random.randint(0, 10, size=5))
    print(np.random.randint(0, 10, size=(2, 3)))

    a = np.random.random((5, 4))
    print(a.shape)
    print(a.shape[0])
    print(a.shape[1])
    print(len(a.shape))
    print(a.size)
    print(a.dtype)

    np.random.seed(693)
    a = np.random.randint(0, 10, size=(5, 4))
    print("Array:\n", a)

    print("Sum of all elements:", a.sum())
    print("Sum of each column:\n", a.sum(axis=0))
    print("Sum of each row:\n", a.sum(axis=1))

    print("Minimum of all elements:", a.min())
    print("Minimum of each column:\n", a.min(axis=0))
    print("Maximum of each row:\n", a.min(axis=1))
    print("Mean of each row:\n", a.mean(axis=1))

    print("Index of max.:", get_max_index(a))

    t1 = time.time()
    print("ML4T")
    t2 = time.time()
    print("The time taken by print statement is", t2 - t1, "seconds")

    nd1 = np.random.random((1000, 10000))

    res_manual, t_manual = how_long(manual_mean, nd1)
    res_numpy, t_numpy = how_long(numpy_mean, nd1)
    print("Manual: {:.6f} ({:.3f} secs.) vs. Numpy: {:.6f} ({:.3f} secs.)".format(res_manual, t_manual,
                                                                                  res_numpy, t_numpy))

    assert abs(res_manual - res_numpy) <= 10e-6, "Result aren't equal!"

    speedup = t_manual / t_numpy
    print("NumPy mean is", speedup, "times faster than manual for loops.")

    a = np.random.randint(0, 10, size=(5, 4))
    print("Array:\n", a)
    print(a[4, 3])
    print(a[0, 1:3])
    print(a[:, 0:3:2])

    a[0, 0] = 1
    print("\nModified(replaced one element):\n", a)

    a[0, :] = 2
    print("\nModified(replaced a row with a single value):\n", a)

    a[:, 3] = [1, 2, 3, 4, 5]
    print("\nModified(replaced a column with a list):\n", a)

    a = np.random.rand(5)
    indices = np.array([1, 1, 2, 3])
    print(a[indices])

    a = np.array([(20, 25, 10, 23, 26, 32, 10, 5, 0), (0, 2, 50, 20, 0, 1, 28, 5, 0)])
    print(a)

    mean = a.mean()
    print(mean)

    print(a[a < mean])
    a[a < mean] = mean
    print(a)

    a = np.array([(1, 2, 3, 4, 5), (10, 20, 30, 40, 50)])
    b = np.array([(100, 200, 300, 400, 500), (1, 2, 3, 4, 5)])
    print("Original array a:\n", a)

    print("\nMultiply a by 2:\n", 2 * a)

    print("\nDivide a by 2:\n", a / 2)
    print("\nDataType of array a:\n", a.dtype)

    print("\nAdd a + b:\n", a + b)

    print("\nMultiply a and b:\n", a * b)

    print("\nDivide a by b:\n", a / b)


if __name__ == "__main__":
    test_run()
