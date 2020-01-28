def fun(x,i):
    beta = 0.2
    gamma = 0.1
    n = 10

    try:
        x_t = x[i-25]
    except IndexError:
        x_t = 0

    return beta*(x_t)/(1+x_t**n) - gamma*x[i]

def EulerForward(x_init,x_end,h):
    N = (x_end-x_init)/h
    x = np.zeros((1,N+1))
    x[0] = x_init
    y[1] = y_init
    for i in range(1,N):
        x[i+1] = x[i]+fun(x,i)
    return x
