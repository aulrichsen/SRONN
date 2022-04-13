import random
import argparse
import matplotlib.pyplot as plt

from Load_Data import get_data

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="Pavia", help="Dataset trained on.")
    parser.add_argument('--rand', action='store_true', help='Make pixel selection random.')

    opt = parser.parse_args()
    arg_str = "Args: " + ', '.join(f'{k}={v}' for k, v in vars(opt).items())
    print(arg_str)

    return opt


if __name__ == "__main__":
    opt = parse_opt()

    _, data, _, _, _, _, _ = get_data(dataset=opt.dataset)

    print(data.shape)

    if opt.rand:
        max_i, max_x, max_y = data.shape[0], data.shape[2], data.shape[3]
        for _ in range(25):
            i, x, y = random.randint(0, max_i-1), random.randint(0, max_x-1), random.randint(0, max_y-1)
            plt.plot(data[i, :, x, y])
    else:
        plt.plot(data[1,:,1,1])
        plt.plot(data[9,:,14,37])
        plt.plot(data[12,:,20,21])
        plt.plot(data[30,:,5,44])
        plt.plot(data[19,:,19,10])
        plt.plot(data[5,:,13,52])
        plt.plot(data[29,:,61,31])
    plt.show()