import numpy as np

# making an AND gate
# x1   x2   b  | out
# -------------------
# 0    0    1  |  0
# 0    1    1  |  0
# 1    0    1  |  0
# 1    1    1  |  1


# b \
#    \
#     \
# x1---sigmoid neuron---->output
#     /
#    /
# x2/


def sigmoid(x, deriv = False):
    if deriv:
        return (x)*(1-x)

    return 1/(1 + np.exp(-x))


# inputs
X = np.array([[0, 0, 1],
              [0, 1, 1],
              [1, 0, 1],
              [1, 1, 1]])

# labels
y = np.array([0, 0, 0, 1])


# weights
np.random.seed(1)
theta = np.array([[0.2, 0.5, .6]])  # 2 * np.random.random((1, 3)) - 1


# learning rate
alpha = .5

for epoch in range(10000):
    # forward propagation
    z = np.dot(theta, X.T)  # weighted sum of all inputs
    a = sigmoid(z)  # activation i.e output of neuron

    # find the error of the outputs
    # E = 0.5 (target - output)^2  =>> squared error function
    E = .5 * np.square(y - a)

    if (epoch % 1000) == 0:
        print("Error at epoch#" + str(epoch) + ": " + str(np.mean(np.abs(E))))

    # back propagation

    partial_E = -(y - a) # partial derivative of Error function wrt 'a'
    partial_a = sigmoid(a, deriv = True) # partial derivative of sigmoid wrt 'z'

    Delta = np.multiply(partial_E, partial_a)  # element-wise multiplication of partial_E and partial_a

    # update weights
    temp = np.dot(Delta, X)
    theta -= alpha * temp


print("Output After Training:")
print(a)

print('\n\n Output rounded:\n')
print(np.rint(a))

print('\n\n Weights at the end: \n')
print(theta)
