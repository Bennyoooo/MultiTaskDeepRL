import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data/')
PLOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../plots/')


def get_section_results(file):
    eval_returns = []
    X = []
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Eval_AverageReturn':
                eval_returns.append(v.simple_value)
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
    return X, eval_returns


if __name__ == "__main__":

    title = 'trained with two tasks'
    file_names = ['box-close_button-press_b3000_box-close-v1_28-10-2020_17-08-45']
    labels = ['']

    for i in range(len(labels)):
        file_path = DATA_PATH + file_names[i]
        _, eval_returns = get_section_results(file_path+"/"+os.listdir(file_path)[0])
        # plot for a single experiment
        x = range(len(eval_returns))

        plt.plot(x, eval_returns, label=labels[i])
        plt.xlabel("num of iterations")
        plt.ylabel("average evaluation return")
    plt.title(title)
    plt.legend()
    plt.savefig(PLOT_PATH + title + '.pdf')
    plt.show()



