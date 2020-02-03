import numpy as np
import sklearn
import sys

def GradientDescent(X, y, stepSize, maxiterations):
    weightVector = np.zeros(X.shape[1])
    weightMatrix = np.zeros((X.shape[1], maxiterations), dtype=float)
    print("Nothing to do yet!")
    return(weightMatrix)

data = None
y = None
epsilon = 0.1
maxiterations = 100
print("sys.argv = " + str(sys.argv))
# TODO: Parsing should be wrapped into a function
#       we want: data, y = parser(fname)
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
    # standardize each column to have μ = 0 and σ = 1
    # this means z_i = (x_i - x_bar) / s, for each column
    # in other words convert all elements to z-scores for each column
    for col in range(temp_ar.shape[1] - 1): # for all but last column (output)
        column = temp_ar[:, col]
        x_bar = np.mean(column)
        s = np.std(column)
        column = (column - x_bar) / s
        temp_ar[:, col] = column
        # this may be numerically unstable but I'm unable to pip3 install scipy=1.3.1 on monsoon
        # scipy version 1.3.1 has scipy.stats.zscore
    np.random.shuffle(temp_ar) # shuffle rows, set of each column remains the same, un-seeded shuffling
    data = temp_ar[:, 0:-1]
    data = data.astype(float)
    y = temp_ar[:, -1]
    y = y.astype(int)
    print("data = " + str(data))
    print("y = " + str(y))
elif sys.argv[1] == 'SAheart.data':
    pass
elif sys.argv[1] == 'zip.train':
    pass

# data is randomly shuffled at this point
num_rows = data.shape[0]
import pdb; pdb.Pdb().set_trace()
train = data[0: int(num_rows * 0.6)]                  # slice of 0% to 60%
test = data[int(num_rows * 0.6): int(num_rows * 0.8)] # slice of 60% to 80%
validation = data[int(num_rows * 0.8):]               # slice of 80% to 100%


GradientDescent(data, y, epsilon, maxiterations)
print("Done!")
