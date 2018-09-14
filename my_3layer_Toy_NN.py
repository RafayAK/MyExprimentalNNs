import numpy as np


# making an XNOR gate
# x1   x2   b  | out
# -------------------
# 0    0    1  |  1
# 0    1    1  |  0
# 1    0    1  |  0
# 1    1    1  |  1

# input_lyr-----hidden_layer----output
# *************************
# b_L1     b_L2
#   \  \       \
#    \  \       \
# x1----a1_Lyr1--a1_Lyr2---->output
#   \  \/         /
#    \ /\        /
#     /  \      /
#    / \  \    /
#   /   \  \  /
# x2----a2_Lyr1


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
y = np.array([1, 0, 0, 1])


# weights
np.random.seed(1)
# theta1 = np.array([[0.5, 0.6, .2], [0.9, .7, .3]])
# theta2 = np.array([[0.7, 0.1, .3]])
mu, sigma = 0, .1 # mean and standard deviation
theta1 = np.random.normal(mu, sigma, (2,3))
theta2 = np.random.normal(mu, sigma, (1,3))

# learning rate
alpha = 10


for epoch in range(50000):
    # forward propagation
    z_Lyr1 = np.dot(theta1, X.T)  # weighted sum of all inputs for Layer1
    a_Lyr1 = sigmoid(z_Lyr1)  # activations for layer
    a_Lyr1 = np.concatenate((a_Lyr1, np.ones((1, a_Lyr1.shape[1]))), axis=0)  # add 1s col for bias units

    z_Lyr2 = np.dot(theta2, a_Lyr1)  # weighted sum of activations from hidden layer
    a_Lyr2 = sigmoid(z_Lyr2)  # Final output

    # find the error of the outputs
    # E = 0.5 (target - output)^2  =>> squared error function
    E = .5 * np.square(y - a_Lyr2)

    if (epoch % 1000) == 0:
        print("Error at epoch#" + str(epoch) + ": " + str(np.mean(np.abs(E))))

    # back propagation
    # gradients for hidden layer
    partial_E = -(y - a_Lyr2)

    partial_a_Lyr2 = sigmoid(a_Lyr2, deriv=True)  # derivative of a_Lyr2 w.r.t to z_Lyr2

    Delta_theta2 = np.dot((partial_E * partial_a_Lyr2), a_Lyr1.T)  # dot product to sum gradients across all examples

    # gradients for input layer
    partial_E_Lyr1 = partial_E * np.dot(partial_a_Lyr2.T, theta2).T
    partial_E_Lyr1 = np.delete(partial_E_Lyr1,-1 , 0) # remove bias row

    partial_a_Lyr1 = sigmoid(a_Lyr1, deriv=True)
    partial_a_Lyr1 = np.delete(partial_a_Lyr1, -1, 0)

    Delta_theta1 = np.dot((partial_E_Lyr1*partial_a_Lyr1),X)

    # update weights
    theta2 -= alpha*Delta_theta2
    theta1 -= alpha*Delta_theta1



print("Output After Training:")
print(a_Lyr2)

print('\n\n Output rounded:\n')
print(np.rint(a_Lyr2))

print('\n\n Weights at the end: \n')
print('\n\n theta1:\n')
print(theta1)

print('\n\n theta2:\n')
print(theta2)

