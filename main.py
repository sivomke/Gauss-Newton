import numpy as np


# find coordinates of a point in N-dim space with Euclidean metric
# given distances to m points with known coordinates


N = 2  # dimension
m = 1000  # number of known points
eps = pow(10, -6)

# square of Euclidean distance in N-dim space
def dist(x, y):
    return np.dot(x-y, x-y)


target = np.array([12, 18])

points = 100*np.random.rand(m, N)  #given points
dist_arr = np.array([dist(target, points[i]) for i in range(m)])  # given distances

# adding Gaussian noise to measurements with the following parameters:
mu = 0  # mean
sigma = 50
dist_arr_disturbed = dist_arr + np.random.normal(mu, sigma, m)
# dist_arr_disturbed = dist_arr + np.random.rand(m)


# residual for i-th point; difference btw calculated and actual distance btw x and points[i]
def r_i(i, x):
    return dist(x, points[i]) - dist_arr_disturbed[i]


def dr_i_dx_j(x, i, j):
    return 2*(x[j] - points[i, j])


def jacobian(x):
    return np.array([[dr_i_dx_j(x, i, j) for j in range(N)] for i in range(m)])


# x_cur = np.array([0, 0])
x_cur = np.random.rand(2)
print("initial guess: {}".format(x_cur))
steps = 0

# for i in range(10):
while True:
    J = jacobian(x_cur)
    J_T = J.transpose()
    inverse = np.linalg.inv(np.matmul(J_T, J))
    r = np.array([r_i(i, x_cur) for i in range(m)])
    delta = np.matmul(inverse, J_T).dot(r)
    print(delta)
    if abs(np.dot(delta, delta))<eps:
        break
    x_cur = x_cur - delta
    steps += 1
    print(x_cur)


print("Number of iterations: {}".format(steps))



