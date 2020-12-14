import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data/')
PLOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../plots/')


def get_section_results(file):
    eval_returns = []
    X = []
    Z = []
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Eval_AverageReturn':
                eval_returns.append(v.simple_value)
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
            elif v.tag == "Eval_StdReturn":
                Z.append(v.simple_value)

    return X, eval_returns, np.array(Z)



if __name__ == "__main__":
    title = 'new_comparison'
    file_names = ['psp_exp1_box-close-v1_13-12-2020_21-46-52', 'nopsp_exp1_box-close-v1_13-12-2020_21-16-27']
    labels = ['psp', 'no_psp']
    plt.figure(figsize=(14, 7))
    for i in range(len(labels)):
        file_path = DATA_PATH + file_names[i]
        _, eval_returns, Z = get_section_results(file_path+"/"+os.listdir(file_path)[0])
        # plot for a single experiment
        x = range(len(eval_returns))

        plt.plot(x, eval_returns, label=labels[i])
        plt.fill_between(x, (eval_returns - Z), (eval_returns + Z),  alpha=0.5)
        plt.xlabel("num of iterations", fontsize=30)
        plt.ylabel("average evaluation return", fontsize=30)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(title)
    plt.legend(fontsize=30)
    plt.savefig(PLOT_PATH + title + '.pdf')
    plt.show()
