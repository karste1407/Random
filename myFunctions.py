import numpy as np
import matplotlib.pyplot as plt

def fun(x,i):
    beta = 0.2
    gamma = 0.1
    n = 10

    try:
        x_t = x[i-25]
    except IndexError:
        x_t = 0

    return beta*(x_t)/(1+x_t**n) - gamma*x[i]

def EulerForward(x_init,t_init,t_end,h):
    N = int((t_end-t_init)/h)
    t = np.linspace(t_init,t_end,N)
    x = np.zeros((N,))
    x[0] = x_init
    for i in range(N-1):
        x[i+1] = x[i]+fun(x,i)
    return t,x

def generateData(verbose=0):
    x0 = 1.5
    t_init = 300
    t_end = 1200
    step = 1
    t,x = EulerForward(x0,t_init,t_end,step)

    if verbose:
        plt.plot(t,x,'b',label='EulerForward')
        plt.xlabel('t')
        plt.ylabel('x(t)')
        plt.title('Mackey-Glass TimeSeries')
        plt.show()

    X_data = np.zeros((5,x.shape[0]))
    shifts = [20,15,10,5,0]
    for t in range(x.shape[0],0):
        for d,shift in enumerate(shifts):
            try:
                X_data[d,t] = x[t-shift]
            except IndexError:
                pass
    return X_data

def generateSubsets(X,nTest,trVal_split):
    X_tmp = X[:,:X.shape[1]-nTest-1]
    X_test = X[:,X.shape[1]-nTest:-1]

    n_train = int(trVal_split*X_tmp.shape[1])
    X_train = X_tmp[:,n_train]
    X_val = X_tmp[:,n_train+1:-1]

    return X_train,X_val,X_test
