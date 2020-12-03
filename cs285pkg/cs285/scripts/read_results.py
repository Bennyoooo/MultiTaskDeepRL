import glob
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def get_section_results(file):
    """
        requires tensorflow==1.12.0
    """
    X = []
    Y = []
    Z = []
    for e in tf.train.summary_iterator(file):
        for v in e.summary.value:
            if v.tag == "Train_EnvstepsSoFar":
                X.append(v.simple_value)
            elif v.tag == "Eval_AverageReturn":
                Y.append(v.simple_value)
            elif v.tag == "Eval_StdReturn":
                Z.append(v.simple_value)
    return np.array(X), np.array(Y), np.array(Z)


if __name__ == "__main__":
    import glob

    # 1a

    fig, ax = plt.subplots()

    logdir = "data/q1_sb_no_rtg_dsa_CartPole-v0_28-09-2020_16-37-38"
    logdir = logdir + "/events*"
    eventfile = glob.glob(logdir)[0]

    X, Y, Z = get_section_results(eventfile)
    x = np.arange(len(X))
    ax.plot(x, Y)
    ax.fill_between(x, (Y - Z), (Y + Z), alpha=0.3, label="sb_no_rtg_dsa")

    logdir = "data/q1_sb_rtg_dsa_CartPole-v0_28-09-2020_16-38-45"
    logdir = logdir + "/events*"
    eventfile = glob.glob(logdir)[0]

    X, Y, Z = get_section_results(eventfile)
    x = np.arange(len(X))
    ax.plot(x, Y)
    ax.fill_between(x, (Y - Z), (Y + Z), alpha=0.3, label="sb_rtg_dsa")

    logdir = "data/q1_sb_rtg_na_CartPole-v0_28-09-2020_16-39-53"
    logdir = logdir + "/events*"
    eventfile = glob.glob(logdir)[0]

    X, Y, Z = get_section_results(eventfile)
    x = np.arange(len(X))
    ax.plot(x, Y)
    ax.fill_between(x, (Y - Z), (Y + Z), alpha=0.3, label="sb_rtg_na")

    plt.xlabel("Iteration Number")
    plt.ylabel("Eval_AverageReturn")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    plt.savefig("q1_sb.png", bbox_inches="tight")

    # 1b

    fig, ax = plt.subplots()

    logdir = "data/q1_lb_no_rtg_dsa_CartPole-v0_28-09-2020_16-40-57"
    logdir = logdir + "/events*"
    eventfile = glob.glob(logdir)[0]

    X, Y, Z = get_section_results(eventfile)
    x = np.arange(len(X))
    ax.plot(x, Y)
    ax.fill_between(x, (Y - Z), (Y + Z), alpha=0.3, label="lb_no_rtg_dsa")

    logdir = "data/q1_lb_rtg_dsa_CartPole-v0_28-09-2020_16-44-53"
    logdir = logdir + "/events*"
    eventfile = glob.glob(logdir)[0]

    X, Y, Z = get_section_results(eventfile)
    x = np.arange(len(X))
    ax.plot(x, Y)
    ax.fill_between(x, (Y - Z), (Y + Z), alpha=0.3, label="lb_rtg_dsa")

    logdir = "data/q1_lb_rtg_na_CartPole-v0_28-09-2020_16-49-13"
    logdir = logdir + "/events*"
    eventfile = glob.glob(logdir)[0]

    X, Y, Z = get_section_results(eventfile)
    x = np.arange(len(X))
    ax.plot(x, Y)
    ax.fill_between(x, (Y - Z), (Y + Z), alpha=0.3, label="lb_rtg_na")

    plt.xlabel("Iteration Number")
    plt.ylabel("Eval_AverageReturn")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    plt.savefig("q1_lb.png", bbox_inches="tight")

    # q2
    fig, ax = plt.subplots()

    logdir = "data/q2_optim_b300_r0.02_InvertedPendulum-v2_28-09-2020_17-17-21"
    logdir = logdir + "/events*"
    eventfile = glob.glob(logdir)[0]

    X, Y, Z = get_section_results(eventfile)
    x = np.arange(len(X))
    ax.plot(x, Y)
    ax.fill_between(x, (Y - Z), (Y + Z), alpha=0.3, label="b300_r0.02")

    plt.xlabel("Iteration Number")
    plt.ylabel("Eval_AverageReturn")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    plt.savefig("q2.png", bbox_inches="tight")

    # q3
    fig, ax = plt.subplots()

    logdir = "data/q3_b40000_r0.005_LunarLanderContinuous-v2_28-09-2020_07-17-04"
    logdir = logdir + "/events*"
    eventfile = glob.glob(logdir)[0]

    X, Y, Z = get_section_results(eventfile)
    x = np.arange(len(X))
    ax.plot(x, Y)
    ax.fill_between(
        x, (Y - Z), (Y + Z), alpha=0.3, label="b40000_r0.005_LunarLanderContinuous-v2",
    )

    plt.xlabel("Iteration Number")
    plt.ylabel("Eval_AverageReturn")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    plt.savefig("q3.png", bbox_inches="tight")

    # q4a

    fig, ax = plt.subplots()

    logdir = "data/q4_search_b10000_lr0.005_rtg_nnbaseline_HalfCheetah-v2_28-09-2020_09-22-06"
    logdir = logdir + "/events*"
    eventfile = glob.glob(logdir)[0]

    X, Y, Z = get_section_results(eventfile)
    x = np.arange(len(X))
    ax.plot(x, Y)
    ax.fill_between(
        x, (Y - Z), (Y + Z), alpha=0.3, label="b10000_lr0.005_rtg_nnbaseline",
    )

    logdir = (
        "data/q4_search_b10000_lr0.01_rtg_nnbaseline_HalfCheetah-v2_28-09-2020_09-31-34"
    )
    logdir = logdir + "/events*"
    eventfile = glob.glob(logdir)[0]

    X, Y, Z = get_section_results(eventfile)
    x = np.arange(len(X))
    ax.plot(x, Y)
    ax.fill_between(
        x, (Y - Z), (Y + Z), alpha=0.3, label="b10000_lr0.01_rtg_nnbaseline",
    )

    logdir = (
        "data/q4_search_b10000_lr0.02_rtg_nnbaseline_HalfCheetah-v2_28-09-2020_09-41-01"
    )
    logdir = logdir + "/events*"
    eventfile = glob.glob(logdir)[0]

    X, Y, Z = get_section_results(eventfile)
    x = np.arange(len(X))
    ax.plot(x, Y)
    ax.fill_between(
        x, (Y - Z), (Y + Z), alpha=0.3, label="b10000_lr0.02_rtg_nnbaseline",
    )

    #######################
    logdir = "data/q4_search_b30000_lr0.005_rtg_nnbaseline_HalfCheetah-v2_28-09-2020_09-50-10"
    logdir = logdir + "/events*"
    eventfile = glob.glob(logdir)[0]

    X, Y, Z = get_section_results(eventfile)
    x = np.arange(len(X))
    ax.plot(x, Y)
    ax.fill_between(
        x, (Y - Z), (Y + Z), alpha=0.3, label="b30000_lr0.005_rtg_nnbaseline",
    )

    logdir = (
        "data/q4_search_b30000_lr0.01_rtg_nnbaseline_HalfCheetah-v2_28-09-2020_10-16-39"
    )
    logdir = logdir + "/events*"
    eventfile = glob.glob(logdir)[0]

    X, Y, Z = get_section_results(eventfile)
    x = np.arange(len(X))
    ax.plot(x, Y)
    ax.fill_between(
        x, (Y - Z), (Y + Z), alpha=0.3, label="b30000_lr0.01_rtg_nnbaseline",
    )

    logdir = (
        "data/q4_search_b30000_lr0.02_rtg_nnbaseline_HalfCheetah-v2_28-09-2020_10-44-25"
    )
    logdir = logdir + "/events*"
    eventfile = glob.glob(logdir)[0]

    X, Y, Z = get_section_results(eventfile)
    x = np.arange(len(X))
    ax.plot(x, Y)
    ax.fill_between(
        x, (Y - Z), (Y + Z), alpha=0.3, label="b30000_lr0.02_rtg_nnbaseline",
    )

    ################################################################

    logdir = "data/q4_search_b50000_lr0.005_rtg_nnbaseline_HalfCheetah-v2_28-09-2020_11-12-07"
    logdir = logdir + "/events*"
    eventfile = glob.glob(logdir)[0]

    X, Y, Z = get_section_results(eventfile)
    x = np.arange(len(X))
    ax.plot(x, Y)
    ax.fill_between(
        x, (Y - Z), (Y + Z), alpha=0.3, label="b50000_lr0.005_rtg_nnbaseline",
    )

    logdir = (
        "data/q4_search_b50000_lr0.01_rtg_nnbaseline_HalfCheetah-v2_28-09-2020_11-58-25"
    )
    logdir = logdir + "/events*"
    eventfile = glob.glob(logdir)[0]

    X, Y, Z = get_section_results(eventfile)
    x = np.arange(len(X))
    ax.plot(x, Y)
    ax.fill_between(
        x, (Y - Z), (Y + Z), alpha=0.3, label="b50000_lr0.01_rtg_nnbaseline",
    )

    logdir = (
        "data/q4_search_b50000_lr0.02_rtg_nnbaseline_HalfCheetah-v2_28-09-2020_12-45-02"
    )
    logdir = logdir + "/events*"
    eventfile = glob.glob(logdir)[0]

    X, Y, Z = get_section_results(eventfile)
    x = np.arange(len(X))
    ax.plot(x, Y)
    ax.fill_between(
        x, (Y - Z), (Y + Z), alpha=0.3, label="b50000_lr0.02_rtg_nnbaseline",
    )

    plt.xlabel("Iteration Number")
    plt.ylabel("Eval_AverageReturn")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    plt.savefig("q4a.png", bbox_inches="tight")

    # q4b
    fig, ax = plt.subplots()

    logdir = "data/q4_b30000_r0.02_HalfCheetah-v2_28-09-2020_19-40-21"
    logdir = logdir + "/events*"
    eventfile = glob.glob(logdir)[0]

    X, Y, Z = get_section_results(eventfile)
    x = np.arange(len(X))
    ax.plot(x, Y)
    ax.fill_between(
        x, (Y - Z), (Y + Z), alpha=0.3, label="b30000_r0.02",
    )

    logdir = "data/q4_b30000_r0.02_nnbaseline_HalfCheetah-v2_28-09-2020_20-31-04"
    logdir = logdir + "/events*"
    eventfile = glob.glob(logdir)[0]

    X, Y, Z = get_section_results(eventfile)
    x = np.arange(len(X))
    ax.plot(x, Y)
    ax.fill_between(
        x, (Y - Z), (Y + Z), alpha=0.3, label="b30000_r0.02_nnbaseline",
    )

    logdir = "data/q4_b30000_r0.02_rtg_HalfCheetah-v2_28-09-2020_20-05-14"
    logdir = logdir + "/events*"
    eventfile = glob.glob(logdir)[0]

    X, Y, Z = get_section_results(eventfile)
    x = np.arange(len(X))
    ax.plot(x, Y)
    ax.fill_between(
        x, (Y - Z), (Y + Z), alpha=0.3, label="b30000_r0.02_rtg",
    )

    logdir = "data/q4_b30000_r0.02_rtg_nnbaseline_HalfCheetah-v2_28-09-2020_20-55-48"
    logdir = logdir + "/events*"
    eventfile = glob.glob(logdir)[0]

    X, Y, Z = get_section_results(eventfile)
    x = np.arange(len(X))
    ax.plot(x, Y)
    ax.fill_between(
        x, (Y - Z), (Y + Z), alpha=0.3, label="b30000_r0.02_rtg_nnbaseline",
    )

    plt.xlabel("Iteration Number")
    plt.ylabel("Eval_AverageReturn")
    plt.legend(bbox_to_anchor=(1.05, 1.0), loc="upper left")
    plt.savefig("q4b.png", bbox_inches="tight")
