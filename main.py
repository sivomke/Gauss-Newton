import numpy as np
import time
from scipy.optimize import least_squares

start_time = time.time()

# find coordinates of a point in N-dim space with Euclidean metric
# given distances to m points with known coordinates
# given distances are disturbed with Gaussian noise

N = 2  # dimension
m = 100  # number of known points
eps = pow(10, -8)


# square of Euclidean distance in N-dim space
def distance(x, y) -> np.array:
    return np.dot(x-y, x-y)


target = np.array([12, 18])
print("target point: {}".format(target))

points = 100*np.random.rand(m, N)  # given points
dist_arr = np.array([distance(target, points[i]) for i in range(m)])  # given distances

# adding Gaussian noise to measurements with the following parameters:
mu = 0  # mean
sigma = 50
dist_arr_disturbed = dist_arr + np.random.normal(mu, sigma, m)


# residual for i-th point
def r_i(i, x):
    return distance(x, points[i]) - dist_arr_disturbed[i]


# squared euclidean norm of error function
# we want to find argmin of f(x)
def f(x):
    return np.sum(np.array([r_i(i, x)**2 for i in range(m)]))


def dr_i_dx_j(x, i, j):
    return 2*(x[j] - points[i, j])


def jacobian(x):
    return np.array([[dr_i_dx_j(x, i, j) for j in range(N)] for i in range(m)])


# function which computes the vector of residuals
def g(x):
    return np.array([r_i(i, x) for i in range(m)])


# custom implementation of Levenberg-Marquardt
def lm():
    x = np.random.rand(2)
    print("initial guess: {}".format(x))
    steps = 0  # counts number of iterations until the solution is found
    damping_factor = .001 # regularization
    eye = np.identity(N)
    while True:
        J = jacobian(x)
        J_T = J.T
        inverse = np.linalg.inv(J_T @ J + damping_factor*eye)
        r = np.array([r_i(i, x) for i in range(m)])
        delta = (inverse @ J_T).dot(r)
        if f(x - delta) < f(x):
            if abs(np.dot(delta, delta)) < eps:
                break
            x = x - delta
            damping_factor *= 0.8
            # steps += 1
        else:
            damping_factor *= 2
        steps += 1
        # print("# {} iteration".format(steps))
        # print("damping factor: {:.6f}".format(damping_factor))
        # print("computed point: {}".format(x))
    return x


res_1 = least_squares(g, np.random.rand(2))
res_2 = lm()
print(res_1.x)
print(res_2)
print(f(res_1.x))
print(f(res_2))
print(f(target))
