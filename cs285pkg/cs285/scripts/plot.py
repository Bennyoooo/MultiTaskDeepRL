import matplotlib
matplotlib.use('Agg')
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
    eval2 = []
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Eval_AverageReturn':
                # eval_returns.append(v.simple_value)
                eval_returns.append(min(1.0, normalize(v.simple_value, -150, 80)))
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
            elif v.tag == "Eval_StdReturn":
                Z.append(v.simple_value)
            elif v.tag == 'Eval_AverageReturn2':
                eval2.append(min(1.0, normalize(v.simple_value, -200, 20000)))

    return X, eval_returns, eval2, np.array(Z)


def normalize(input, min, max):
    return (input - min)/float(max - min)


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


if __name__ == "__main__":


    # title = 'psp_two_task_normalized_new'
    # file_names = ['psp_twotask_box-close-v1_14-12-2020_17-52-07']
    # labels = ['']

    title = 'psp_complex2'
    file_names = ['psp_conplex_s128_buttonpress_box-close-v1_14-12-2020_20-48-18']
    labels = ['']

    # title = 'no_psp_s128'
    # file_names = ['s128_nopsp_box-close-v1_14-12-2020_17-06-44']
    # labels = ['']

    #three comparison:
    # title = 'large_net_large_batch'
    # file_names = ['s128_nopsp_box-close-v1_15-12-2020_21-29-43',
    #               'psp-layer4size128batch7000_box-close-v1_15-12-2020_05-40-25']
    # labels = ['no_psp', 'psp']

    # title = 'large_net_large_batch_second2'
    # file_names = ['s128_nopsp_box-close-v1_15-12-2020_21-29-43',
    #               'psp-layer4size128batch7000_box-close-v1_15-12-2020_05-40-25']
    # labels = ['no_psp', 'psp']

    # title = 'second_task_s64'
    # file_names = ['no_psp_s64_buttonpress_box-close-v1_15-12-2020_21-12-45',
    #               'psp_twotask_box-close-v1_14-12-2020_17-52-07']
    # labels = ['no_psp', 'psp']

    plt.figure(figsize=(14, 7))
    for i in range(len(labels)):
        file_path = DATA_PATH + file_names[i]
        _, eval_returns, evals2, Z = get_section_results(file_path+"/"+os.listdir(file_path)[0])
        # plot for a single experiment
        x = range(len(eval_returns))

        plt.plot(x, smooth(eval_returns, 3), label='1')
        plt.plot(x, smooth(evals2, 10), label='2')
        # plt.fill_between(x, (eval_returns - Z), (eval_returns + Z),  alpha=0.5)
        plt.xlabel("num of iterations", fontsize=30)
        plt.ylabel("average evaluation return", fontsize=30)
        plt.axvline(x=150, color='purple', linestyle='--')
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.title(title)
    plt.legend(fontsize=30)
    plt.savefig(PLOT_PATH + title + '.png')
    plt.show()



