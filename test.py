import numpy as np

def GradientDescent(X, y, stepSize, maxiterations):
    weightVector = np.zeros(X.shape[1])
    weightMatrix = np.zeros((X.shape[1], maxiterations), dtype=float)
    print("Nothing to do yet!")
    return(weightMatrix)

data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
y = np.array([0, 1])
epsilon = 0.2
maxiterations = 100
GradientDescent(data, y, epsilon, maxiterations)
print("Done!")
