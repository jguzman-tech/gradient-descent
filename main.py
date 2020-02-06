import numpy as np
import sklearn
from sklearn import metrics
import sys
from scipy import stats
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def MeanLogisticLoss(weightMatrix, X, y):
    y_tilde = np.copy(y)
    y_tilde[y_tilde != 1] = -1
    result = []
    m = X.shape[0]
    maxiter = weightMatrix.shape[1]
    for col in range(maxiter):
        theta = weightMatrix[:, col][np.newaxis].T # n x 1 matrix
        mySum = 0.0
        for row in range(m):
            mySum += np.log(1 + np.exp(-1 * y_tilde[row, 0] * np.matmul(X[row][np.newaxis], theta)[0, 0]))
        result.append(mySum / m)
    return result

def Gradient(theta, X, y_tilde):
    result = np.zeros(X.shape[1], dtype=float)
    m = X.shape[0]
    for row in range(m): # for each row
        numerator = -1 * y_tilde[row, 0] * X[row]
        denominator = (1 + np.exp(y_tilde[row, 0] * np.matmul(theta[np.newaxis], X[row][np.newaxis].T)))
        result += (numerator / denominator)[0, :] # numerator is n x 1, denominator is 1 x 1
    result = result / m
    return result

def GradientDescent(X, y, stepSize, maxiterations):
    weightVector = np.zeros(X.shape[1], dtype=float)
    weightMatrix = np.zeros((X.shape[1], maxiterations), dtype=float)
    y_tilde = np.copy(y)
    y_tilde[y_tilde != 1] = -1
    for k in range(1, maxiterations + 1):
        newWeightVector = weightVector - stepSize * Gradient(weightVector, X, y_tilde)
        weightVector = newWeightVector
        weightMatrix[:, k - 1] = newWeightVector
    return weightMatrix

def parse(fname, seed):
    if(fname == 'spam.data'):
        all_rows = []
        with open('spam.data') as fp:
            for line in fp:
                row = line.split(' ')
                all_rows.append(row)
        temp_ar = np.array(all_rows, dtype=float)
        temp_ar = temp_ar.astype(float)
        # standardize each column to have μ = 0 and σ^(2) = 1
        # in other words convert all elements to z-scores for each column
        for col in range(temp_ar.shape[1] - 1): # for all but last column (output)
            temp_ar[:, col] = stats.zscore(temp_ar[:, col])
        np.random.seed(seed)
        np.random.shuffle(temp_ar) # shuffle rows, set of columns remain the same
    elif(fname == 'SAheart.data'):
        # Make sure to skip the first row
        # Make sure to replace the present and absent strings
        # Get the zscore if you want to be complete
        # Drop the row column
        all_rows = []
        with open('SAheart.data') as fp:
            for line in fp:
                row = line.split(',')
                all_rows.append(row)
        all_rows = all_rows[1:]
        all_rows=np.array(all_rows)
        all_rows[all_rows == "Present"] = "1"
        all_rows[all_rows == "Absent"] = "0"
        all_rows= all_rows[:,1:]
        temp_ar = np.array(all_rows, dtype=float)
        # standardize each column to have μ = 0 and σ^(2) = 1
        # in other words convert all elements to z-scores for each column
        for col in range(temp_ar.shape[1] - 1): # for all but last column (output)
            temp_ar[:, col] = stats.zscore(temp_ar[:, col])
        np.random.seed(seed)
        np.random.shuffle(temp_ar) # shuffle rows, set of columns remain the same
    elif(fname == 'zip.train'):
        all_rows = []
        with open('zip.train') as fp:
            for line in fp:
                line= line.strip()
                row = line.split(' ')
                all_rows.append(row)
        all_rows=np.array(all_rows)
        all_rows=all_rows[(all_rows[:,0] == "0.0000") |  (all_rows[:,0] == "1.0000")]
        all_rows[:,[0,256]]= all_rows[:,[256,0]]
        temp_ar = np.array(all_rows, dtype=float)
        # standardize each column to have μ = 0 and σ^(2) = 1
        # in other words convert all elements to z-scores for each column
        for col in range(temp_ar.shape[1] - 1): # for all but last column (output)
            temp_ar[:, col] = stats.zscore(temp_ar[:, col])
        np.random.seed(seed)
        np.random.shuffle(temp_ar) # shuffle rows, set of columns remain the same
    else:
        raise Exception("Unknown dataset")
    return temp_ar

print("sys.argv = " + str(sys.argv))
# TODO: Parsing should be wrapped into a function
#       we want: temp_ar = parse(fname)
if len(sys.argv) < 5 or sys.argv[1] not in {'spam.data', 'SAheart.data', 'zip.train'}:
    help_str = """Execution example: python3 main.py <dataset> <stepSize> <maxiterations> <seed>
The valid dataset values are: spam.data, SAheart.data, and zip.train.
stepSize must be a float
maxiterations must be an int
seed must be an int
"""
    print(help_str)
    exit(0)

stepSize = float(sys.argv[2])
maxiterations = int(sys.argv[3])
seed = int(sys.argv[4])
temp_ar = parse(sys.argv[1], seed)

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
print('  {0: >10} {1: >4} {2: >4}'.format('set', '0', '1'))
print('  {0: >10} {1: >4} {2: >4}'.format('test',
                                          str((test_y == 0).sum()),
                                          str((test_y == 1).sum())))
print('  {0: >10} {1: >4} {2: >4}'.format('train',
                                          str((train_y == 0).sum()),
                                          str((train_y == 1).sum())))
print('  {0: >10} {1: >4} {2: >4}'.format('validation',
                                          str((validation_y == 0).sum()),
                                          str((validation_y == 1).sum())))

weightMatrix = GradientDescent(train_X, train_y, stepSize, maxiterations)

validation_predict = np.matmul(validation_X, weightMatrix)
validation_predict[validation_predict >= 0.0] = 1
validation_predict[validation_predict < 0.0] = 0

train_predict = np.matmul(train_X, weightMatrix)
train_predict[train_predict >= 0.0] = 1
train_predict[train_predict < 0.0] = 0

validation_error = []
train_error = []
for i in range(maxiterations):
    validation_error.append(100 * (np.mean(validation_y[:, 0] != validation_predict[:, i])))
for i in range(maxiterations):
    train_error.append(100 * (np.mean(train_y[:, 0] != train_predict[:, i])))

fnames = []
# % error plot
plt.plot(validation_error, c="red", label="validation")
plt.scatter([np.where(validation_error == np.min(validation_error))[0][0]], [np.min(validation_error)], marker='o', s=80, facecolors='none', edgecolors='r')
plt.plot(train_error, c="blue", label="train")
plt.scatter([np.where(train_error == np.min(train_error))[0][0]], [np.min(train_error)], marker='o', s=80, facecolors='none', edgecolors='b')
plt.xlabel('Iteration')
plt.ylabel('% Error')
plt.legend()
fname = sys.argv[1] + "_step_" + str(stepSize) + "_itr_" + str(maxiterations) + "_seed_" + str(seed) + "_err_plot.png"
fname="./figures/" + fname
plt.savefig(fname)
plt.clf()
fnames.append(fname)

validation_mll = MeanLogisticLoss(weightMatrix, validation_X, validation_y)
train_mll = MeanLogisticLoss(weightMatrix, train_X, train_y)
# Logistic Loss plot
plt.plot(validation_mll, c="red", label="validation")
plt.scatter([np.where(validation_mll == np.min(validation_mll))[0][0]], [np.min(validation_mll)], marker='o', s=80, facecolors='none', edgecolors='r')
plt.plot(train_mll, c="blue", label="train")
plt.scatter([np.where(train_mll == np.min(train_mll))[0][0]], [np.min(train_mll)], marker='o', s=80, facecolors='none', edgecolors='b')
plt.xlabel('Iteration')
plt.ylabel('Logistic Loss')
plt.legend()
fname = sys.argv[1] + "_step_" + str(stepSize) + "_itr_" + str(maxiterations) + "_seed_" + str(seed) + "_mll_plot.png"
fname="./figures/" + fname
plt.savefig(fname)
plt.clf()
fnames.append(fname)

# The sklearn.metrics.roc_curve automatically converts our real numbers to probabilities
# We make roc_curves with the test set, an using the iteration # with the minimum LL for
# our validaiton set
optimal_itr = np.where(validation_mll == np.min(validation_mll))[0][0]
test_predict = np.matmul(test_X, weightMatrix)
fpr, tpr, thresholds = metrics.roc_curve(test_y[:, 0], test_predict[:, optimal_itr], pos_label=1)
linear = np.linspace(0, 1, 1000)
plt.plot(linear, linear, linestyle='--', color="black")
plt.plot(fpr, tpr, color='green')
plt.xlabel('FPR')
plt.ylabel('TPR')
fname = sys.argv[1] + "_step_" + str(stepSize) + "_itr_" + str(maxiterations) + "_seed_" + str(seed) + "_roc_curve.png"
fname="./figures/" + fname
plt.savefig(fname)
plt.clf()
fnames.append(fname)

print("Figures Created ")
for i in range(len(fnames)):
    print(str(fnames[i]))
