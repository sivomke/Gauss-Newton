import numpy as np


# find coordinates of a point in N-dim space with Euclidean metric
# given distances to m points with known coordinates

N = 2  # dimension
m = 10  # number of known points


# square of Euclidean distance in N-dim space
def dist(x, y):
    return np.dot(x-y, x-y)


target = np.array([10, 10])

points = 100*np.random.rand(m, N)  #given points
dist_arr = np.array([dist(target, points[i]) for i in range(m)])  # given distances


# residual for i-th point; difference btw calculated and actual distance btw x and points[i]
def r_i(i, x):
    return dist(x, points[i]) - dist_arr[i]


def dr_i_dx_j(x, i, j):
    return 2*(x[j] - points[i, j])


def jacobian(x):
    return np.array([[dr_i_dx_j(x, i, j) for j in range(N)] for i in range(m)])


x_cur = np.array([0, 0])

for i in range(10):
    J = jacobian(x_cur)
    J_T = J.transpose()
    inverse = np.linalg.inv(np.matmul(J_T, J))
    r = np.array([r_i(i, x_cur) for i in range(m)])
    x_cur = x_cur - np.matmul(inverse, J_T).dot(r)
    print(x_cur)





