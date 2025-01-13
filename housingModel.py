# Mikail Usman
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Setting up data (training and testing sets)
df = pd.read_csv('Housing/train.csv', sep=',')

# Singling out the headings we need as features to predict the price
featureColumns = ['LotFrontage', 'LotArea', 'OverallQual', 'OverallCond',
    'YearBuilt', 'BsmtFinSF1', 'BsmtUnfSF', 'TotalBsmtSF', '1stFlrSF',
    '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'FullBath',
    'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd',
    'Fireplaces', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF',
    'EnclosedPorch', 'MoSold']

def reportMetrics():
    """
    Analyzing the training set and reporting the required metrics.
    """
    priceColumn = df['Price']
    print(f'Records: {np.count_nonzero(priceColumn)}') # How many records are there in the training set
    print(f'Mean: {np.mean(priceColumn)}') # What is the mean value of the price
    print(f'Min: {np.min(priceColumn)}') # Calculating minimum value 
    print(f'Max: {np.max(priceColumn)}') # Calculating maximum value
    print(f'Standard Deviation: {np.std(priceColumn)}') # Calculating standard deviation

def plotHistogram():
    """
    Plotting a histogram for the sales price.
    """
    priceColumn = df['Price']
    plt.figure(figsize=(10, 6)) # Setting the size of the figure
    plt.hist(priceColumn, bins=40) # Creating the histogram itself (bins = bars = ranges)
    plt.title('Sales Price Histogram') # Setting up the title
    plt.xlabel('Prices')
    plt.ylabel('Frequency')
    plt.grid(True) # Enabling grid
    plt.show()

def plotScatter():
    """
    Plotting a pair-wise scatter plot of the following features: GrLivArea, BedroomAbvGr, TotalBsmtSF, FullBath
    """
    features = ['GrLivArea', 'BedroomAbvGr', 'TotalBsmtSF', 'FullBath'] # Pairwise columns
    fig, axes = plt.subplots(nrows=3, ncols=3, figsize=(12, 12)) # Creating a figure with 6 plots for each pair
    axes = axes.flatten() # Flatten the axes for easy iteration
    plotCounter = 0 # Initialize a plot counter

    # Create scatter plots for each unique pair of features
    for i, feature1 in enumerate(features):
        for j, feature2 in enumerate(features):
            if i < j:  # Avoiding identical plots
                ax = axes[plotCounter]
                ax.scatter(df[feature1], df[feature2], alpha=0.5, color='skyblue', edgecolors='black') # Edge = boundary/border
                ax.set_xlabel(feature1)
                ax.set_ylabel(feature2)
                ax.set_title(f'{feature1} vs {feature2}')
                plotCounter += 1

    # Hide unused subplots (for the empty spots)
    for i in range(plotCounter, len(axes)):
        axes[i].axis('off')

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()

def pred(w, X):
    """
    Calculates the predicted value of the price based on 
    the current weights (w) and feature values (x).
    """
    x = df[X].values # Accessing only the values of the selected headings (featureColumns)
    # yPred is a matrix of shape (818, 1) -> 818 is the total no. of value rows for train.csv
    yPred = np.dot(x, w) # Calculating the dot product of the featureColumn values (x) and the weight matrix (w)
    
    return yPred

def loss(yPred):
    """
    Calculates the loss based on the difference between predicted sale price and the correct
    sale price. Implements the mean squared error of the difference as the loss function.
    """
    price = (df['Price'] - df['Price'].mean()) / df['Price'].std() # Normalizng the target price to help training.
    yTrue = price.values # Accessing true Price values for the houses
    diffSquared = np.square(yPred - yTrue) # Calculating the squared differences
    mse = np.mean(diffSquared) # Calculating the mean of the squared differences (MSE)

    return mse

def gradient(yPred, X):
    """
    Calculates the gradient of loss function based on the predicted price and the correct price.
    """
    yPred = yPred.reshape(-1, 1)

    x = df[X].values # Accessing values from the featureColumn (values for all features)
    price = (df['Price'] - df['Price'].mean()) / df['Price'].std()
    yTrue = price.values.reshape(-1, 1) # Accessing true Price values for the houses
    diff = yPred - yTrue # Calculating the difference between predicted and actual sales price.
    gradMSE = (2/yTrue.shape[0]) * np.dot(x.T, diff) # Calculating the gradient of loss function (Shape: (n_features, 1))

    return gradMSE

def update(grad, w, a):
    """
    Gradient Descent -> Updates weights based on the gradient.
    Uses 'alpha': learning rate (a).
    Needs to be called within an optimization loop where the weights are updated iteratively.
    """
    w = w - a * grad # Updates the weights using the formula: Wt+1 = Wt - α * gradient

    return w

def trainModel(a, i):
    """
    Algorithm to train the model to predict housing prices by iteratively updating weights.
    """
    mseTable = [] # Storing the MSE values for each iteration
    # Smaller initial weights to prevent slower learning
    w = np.random.random((25, 1)) * 0.01 # Initializing random weights matrix (row, col) 
    # Loop to update weights
    for value in range(i):
        Y = pred(w, featureColumns)
        MSE = loss(Y)
        mseTable.append(MSE)
        Delta = gradient(Y, featureColumns)
        w = update(Delta, w, a)

    return mseTable

def compareAlphas():
    """
    Setting α = 10−11 and α = 10−12 to report a learning curve
    and analyze the effects of different alpha values to the learning curve and MSE.
    """
    # Running model with specific alpha values
    iterations = 200000
    alpha1 = 0.2
    alpha2 = 1e-11
    alpha3 = 1e-12
    mse1 = trainModel(alpha2, iterations)
    mse2 = trainModel(alpha3, iterations)

    # Plotting findings
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(mse1)), mse1, label=f"α = {alpha2}", color='blue')
    plt.plot(range(len(mse2)), mse2, label=f"α = {alpha3}", color='red')
    plt.title("Learning Curve for Model Training")
    plt.xlabel("Iterations")
    plt.ylabel("MSE")
    plt.legend()
    plt.grid(True)
    plt.show()

def predictPrice():
    """
    Predicting the price using the trained weight for the test.csv file.
    """
    tf = pd.read_csv('Housing/test.csv', sep=',') # Accessing test.csv
    w = np.random.random((25, 1)) * 0.01 # Initializing random weights matrix (row, col) 
    testValues = tf[featureColumns].values  # Making a feature matrix based on test.csv data
    testPrices = tf['Price'].values # Accessing true test.csv prices
    a = 1e-11
    iterations = 37500
    for value in range(iterations):
        Y = pred(w, featureColumns)
        MSE = loss(Y)
        Delta = gradient(Y, featureColumns)
        w = update(Delta, w, a)

    # Predicting the price - Finding ^Y = SUM (w*x) 
    predictions = np.dot(testValues, w) # Dot product of weights, features
    testMSE = np.mean(np.square(predictions - testPrices)) # Calculating the MSE for the predicted values
    return predictions, MSE, testMSE

if __name__ == "__main__":
    print(predictPrice())