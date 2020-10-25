import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

DATA_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data/')
PLOT_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../plots/')

def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    X = []
    Y = []
    Z = []
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == 'Train_EnvstepsSoFar':
                X.append(v.simple_value)
            elif v.tag == 'Train_AverageReturn':
                Y.append(v.simple_value)
            elif v.tag == 'Train_BestReturn':
                Z.append(v.simple_value)
    return X, Y, Z

if __name__ == '__main__':

    title = 'experiment_1_MsPacman-v0'
    file_names = ['hw3_q1_MsPacman-v0_11-10-2020_21-49-07']
    labels = ['']

    # title = 'experiment_2'
    # file_names = [['hw3_q2_dqn_1_LunarLander-v3_11-10-2020_19-00-45',
    #               'hw3_q2_dqn_2_LunarLander-v3_11-10-2020_19-50-53',
    #               'hw3_q2_dqn_3 _LunarLander-v3_12-10-2020_07-02-10'],
    #               ['hw3_q2_doubledqn_1_LunarLander-v3_11-10-2020_20-53-54',
    #                'hw3_q2_doubledqn_2 _LunarLander-v3_12-10-2020_08-32-04',
    #                'hw3_q2_doubledqn_3 _LunarLander-v3_12-10-2020_09-10-20'
    #               ]]
    # labels = ['dqn', 'double_dqn']

    # plot for experiment 1b
    # title = 'experiment_4_CartPole-v0'
    # file_names = ['hw3_ q4_1_100_CartPole-v0_18-10-2020_10-22-53',
    #               'hw3_ q4_10_10_CartPole-v0_18-10-2020_10-25-51',
    #               'hw3_ q4_100_1_CartPole-v0_18-10-2020_10-24-29']
    # labels = ['ntu:1_ngsptu:100', 'ntu:10_ngsptu:10', 'ntu:100_ngsptu:1']

    # title = 'experiment_2'
    # file_names = ['q3_b40000_r0.005_LunarLanderContinuous-v2_29-09-2020_17-34-48']
    # labels = ['']

    # title = 'experiment_5_HalfCheetah-v2'
    # file_names = ['hw3_ q5_10_10_HalfCheetah-v2_18-10-2020_10-33-30']
    # labels = ['']

    # title = 'experiment_4a_HalCheetah_search'
    # file_names = ['q4_search_b10000_lr0.005_rtg_nnbaseline_HalfCheetah-v2_29-09-2020_04-35-28',
    #               'q4_search_b10000_lr0.01_rtg_nnbaseline_HalfCheetah-v2_29-09-2020_08-31-08',
    #               'q4_search_b10000_lr0.02_rtg_nnbaseline_HalfCheetah-v2_29-09-2020_08-46-58',
    #               'q4_search_b30000_lr0.005_rtg_nnbaseline_HalfCheetah-v2_29-09-2020_07-19-15',
    #               'q4_search_b30000_lr0.01_rtg_nnbaseline_HalfCheetah-v2_29-09-2020_04-57-58',
    #               'q4_search_b30000_lr0.02_rtg_nnbaseline_HalfCheetah-v2_29-09-2020_07-38-56',
    #               'q4_search_b50000_lr0.005_rtg_nnbaseline_HalfCheetah-v2_29-09-2020_06-44-40',
    #               'q4_search_b50000_lr0.01_rtg_nnbaseline_HalfCheetah-v2_29-09-2020_05-37-54',
    #               'q4_search_b50000_lr0.02_rtg_nnbaseline_HalfCheetah-v2_29-09-2020_06-08-12']
    # labels = ['b:10000_lr:0.005', 'b:10000_lr:0.01', 'b:10000_lr:0.02','b:30000_lr:0.005', 'b:30000_lr:0.01',
    #           'b:30000_lr:0.02', 'b:50000_lr:0.005', 'b:50000_lr:0.01', 'b:50000_lr:0.02']

    # plot for experiment 1b
    # title = 'experiment_3_LunarLander-v3_'
    # file_names = ['hw3_q2_dqn_1_LunarLander-v3_11-10-2020_19-00-45',
    #               'hw3_q3_hparam1_LunarLander-v3_18-10-2020_19-24-20',
    #               'hw3_q3_hparam2_LunarLander-v3_18-10-2020_20-02-46',
    #               'hw3_q3_hparam3_LunarLander-v3_18-10-2020_21-08-31']
    # labels = ['rl:1e-3', 'lr:1e-2', 'lr:1e-4', 'lr:5e-2']

    for i in range(len(labels)):
        # returns = []
        # for j in range(3):
        #     file_path = DATA_PATH + file_names[i][j]
        #     x, eval_returns = get_section_results(file_path + "/" + os.listdir(file_path)[0])
        #     returns.append(eval_returns)
        # eval_returns = np.sum(returns, axis=0)/3
        file_path = DATA_PATH + file_names[i]
        x, eval_returns, z = get_section_results(file_path + "/" + os.listdir(file_path)[0])

        plt.plot(x[1:], eval_returns, label='average')
        plt.plot(x[2:], z, label='best so far')
        plt.xlabel("num of steps")
        plt.ylabel("average train_returns")
    plt.title(title)
    plt.legend()
    plt.savefig(PLOT_PATH + title + '.pdf')
    plt.show()
