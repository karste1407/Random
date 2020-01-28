def generateData():
    from myFunctions import EulerForward
    x0 = 1.5
    t_init = 300
    t_end = 1500
    x = EulerForward(x0,t-init,t_end,step)
    return x


def main():
    x = generateData():
    print(x)

if __name__ == "__main__":
    main()
