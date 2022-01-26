import matplotlib.pyplot as plt
import re


def read(name):
    f = open(name, 'r')
    file = f.read()
    file = re.sub('\\[', '', file)
    file = re.sub('\\]', '', file)
    f.close()

    return [float(i) for idx, i in enumerate(file.split(','))]


def draw():
    train = read('./result/train_loss.txt')
    test = read('./result/valid_loss.txt')
    plt.plot(train, 'r', label='train')
    plt.plot(test, 'b', label='validation')
    plt.legend(loc='upper right')
    plt.xlabel('epoch')
    plt.ylabel("loss")
    plt.title('training result')
    plt.grid(True, which='both', axis='both')
    plt.savefig("./result/loss_graph.png")


if __name__ == '__main__':
    draw()
