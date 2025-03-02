import matplotlib.pyplot as plt

def plot_counts(count_logs: dict) -> None:
    
    param = list(count_logs.keys())
    val = list(count_logs.values())
    
    xmin = 0
    xmax = max(val) + 1
    tick_div = (xmax - xmin) // 10

    xlist = range(xmin, xmax, tick_div)

    fig = plt.figure(figsize = (15, 7))

    plt.barh(y = param, width = val)
    plt.xticks(xlist)

    plt.xlabel("Count of Operations")
    plt.ylabel("Augmentation")

    plt.title("Augmentation Operations Done on the Images")

    plt.savefig('results.png')

    plt.show()

    