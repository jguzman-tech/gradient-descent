import numpy as np
import sklearn
import sys
from scipy import stats
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def LogisticLoss(theta, X, y):
    result = 0.0
    m = X.shape[0]
    for row in range(m): # for each row
        y_tilde = -1
        if(y[row, 0] == 1):
            y_tilde = 1
        numerator = -1 * y_tilde * X[row]
        denominator = (1 + np.exp(y_tilde * theta.T * X[row]))
        result += numerator / denominator
    result = result / m
    return result

def GradientDescent(X, y, stepSize, maxiterations):
    weightVector = np.zeros(X.shape[1], dtype=float)
    weightMatrix = np.zeros((X.shape[1], maxiterations), dtype=float)
    for k in range(1, maxiterations + 1):
        newWeightVector = weightVector - stepSize * LogisticLoss(weightVector, X, y)
        weightVector = newWeightVector
        weightMatrix[:, k - 1] = newWeightVector
    return weightMatrix

X = None
y = None
epsilon = float(sys.argv[2])
maxiterations = int(500)
seed = int(50)
print("sys.argv = " + str(sys.argv))
# TODO: Parsing should be wrapped into a function
#       we want: X, y = parser(fname)
if len(sys.argv) < 2 or sys.argv[1] not in {'spam.data', 'SAheart.data', 'zip.train'}:
    print("Execution example: main.py <dataset>")
    print("The valid dataset values are: spam.data, SAheart.data, and zip.train.")
    exit(0)
elif sys.argv[1] == 'spam.data':
    all_rows = []
    with open('spam.data') as fp:
        for line in fp:
            row = line.split(' ')
            all_rows.append(row)
    temp_ar = np.array(all_rows, dtype=float)
    # standardize each column to have μ = 0 and σ^(2) = 1
    # in other words convert all elements to z-scores for each column
    for col in range(temp_ar.shape[1] - 1): # for all but last column (output)
        temp_ar[:, col] = stats.zscore(temp_ar[:, col])
    np.random.seed(seed)
    np.random.shuffle(temp_ar) # shuffle rows, set of columns remain the same
elif sys.argv[1] == 'SAheart.data':
    # Make sure to replace the present and absent strings
    # Get the zscore if you want to be complete
    # Drop the row column
    pass
elif sys.argv[1] == 'zip.train':
    pass

# temp_ar is randomly shuffled at this point
num_rows = temp_ar.shape[0]

X = temp_ar[:, 0:-1] # m x n
X = X.astype(float)
y = np.array([temp_ar[:, -1]]).T # make it a row vector, m x 1
y = y.astype(int)

train_X = X[0: int(num_rows * 0.6)]                        # slice of 0% to 60%
train_y = y[0: int(num_rows * 0.6)]                        # slice of 0% to 60%
test_X = X[int(num_rows * 0.6): int(num_rows * 0.8)]       # slice of 60% to 80%
test_y = y[int(num_rows * 0.6): int(num_rows * 0.8)]       # slice of 60% to 80%
validation_X = X[int(num_rows * 0.8):]                     # slice of 80% to 100%
validation_y = y[int(num_rows * 0.8):]                     # slice of 80% to 100%

print('            y')
print('set       0     1')
print('  {0: >10} {1: >4} {2: >4}'.format('test',
                                          str((test_y == 1).sum()),
                                          str((test_y == 0).sum())))
print('  {0: >10} {1: >4} {2: >4}'.format('train',
                                          str((train_y == 1).sum()),
                                          str((train_y == 0).sum())))
print('  {0: >10} {1: >4} {2: >4}'.format('validation',
                                          str((validation_y == 1).sum()),
                                          str((validation_y == 0).sum())))
weightMatrix = GradientDescent(train_X, train_y, epsilon, maxiterations)

validation_predict = np.matmul(validation_X, weightMatrix)
validation_predict[validation_predict < 0.0] = 0
validation_predict[validation_predict > 0.0] = 1

train_predict = np.matmul(train_X, weightMatrix)
train_predict[train_predict < 0.0] = 0
train_predict[train_predict > 0.0] = 1

validation_error = []
train_error = []
for i in range(maxiterations):
    validation_error.append(100 * (np.mean(validation_y[:, 0] != validation_predict[:, i])))
for i in range(maxiterations):
    train_error.append(100 * (np.mean(train_y[:, 0] != train_predict[:, i])))

plt.plot(validation_error, c="red", label="validation")
plt.plot(train_error, c="blue", label="train")
plt.xlabel('% Error')
plt.ylabel('Iteration')
plt.savefig(sys.argv[1] + "_itr_" + str(maxiterations)+ "_step_" + str(epsilon) + "_seed_" + str(seed) + "plot.png")
    
print("Done!")
# import pdb; pdb.Pdb().set_trace() # break into pdb
