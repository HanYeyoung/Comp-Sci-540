import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
def linear_regression(filename):
    df = pd.read_csv(filename) #read the original dataset

    plt.plot(df['year'], df['days'])
    plt.xlabel("Year")
    plt.ylabel("Number of frozen days")
    plt.savefig("plot.jpg") # save the plot

    data = np.array(df['year'])
    ones = np.ones(len(data)).astype(int)

    # Represent the data as a matrix X
    X = np.column_stack((ones, data))
    print("Q3a:")
    print(X)

    Y = np.array(df['days'])
    print("Q3b:")
    print(Y)

    Z = np.dot(X.T,X)
    print("Q3c:")
    print(Z)

    I = np.linalg.inv(Z)
    print("Q3d:")
    print(I)

    PI = I @ X.T
    print("Q3e:")
    print(PI)

    hat_beta = PI@Y
    print("Q3f:")
    print(hat_beta)

    # predict the number of ice days for winter 2022-23
    x_test = 2022 # test item
    # check to see how close your prediction was
    y_hat_test = hat_beta[0] + hat_beta[1] * x_test
    print("Q4: " + str(y_hat_test))

    # Interpret the meaning of the sign for Mendota ices
    if hat_beta[1] > 0:
        sign = ">"
        short_answer = "The number of frozen days increases, winters get colder over time."
    elif hat_beta[1] < 0:
        sign = "<"
        short_answer = "The number of frozen days decreases, winters get warmer over time."
    else:
        sign = "="
        short_answer = "Winters remain consistent over the years."

    print("Q5a: " + sign)
    print("Q5b: " + short_answer) #explain all three signs regarding Mendota ice in general.

    x_star = -hat_beta[0] / hat_beta[1] # predict the year x_star by which Lake Mendota will no longer freeze
    print("Q6a: " + str(x_star))

    max_year = df['year'].max()
    min_year = df['year'].min()

    # discuss whether x_star is a compelling prediction based on the trends in the data
    if min_year <= x_star <= max_year:
        answer = "The prediction is within the dataset's range; Lake Mendota might stop freezing in the near future."
    elif x_star < min_year:
        answer = "The prediction is before the dataset's range; seems impossible"
    else:
        answer = "The prediction might be or might not be compelling"
    print("Q6b: " + answer)

if __name__ == "__main__":
    filename = sys.argv[1] # the first argument as string
    linear_regression(filename)