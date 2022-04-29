from collections import OrderedDict
import pickle
import os
import sys
import time

import gym
from gym import wrappers
import numpy as np
import torch
from cs285.infrastructure import pytorch_util as ptu
import metaworld
import random

from cs285.infrastructure import utils
from cs285.infrastructure.logger import Logger

# how many rollouts to save as videos to tensorboard
MAX_NVIDEO = 2
MAX_VIDEO_LEN = 40  # we overwrite this in the code below


class RL_Trainer(object):
    def __init__(self, params):

        #############
        ## INIT
        #############

        # Get params, create logger
        self.params = params
        self.logger = Logger(self.params["logdir"])

        self.second_task = False
        self.period = params['period']

        # Set random seeds
        seed = self.params["seed"]
        np.random.seed(seed)
        torch.manual_seed(seed)
        ptu.init_gpu(use_gpu=not self.params["no_gpu"], gpu_id=self.params["which_gpu"])

        #############
        ## ENV
        #############

        # Make the gym environment
        mt1 = metaworld.MT1(self.params['env_name'])
        self.env = mt1.train_classes[self.params['env_name']]()
        task = random.choice(mt1.train_tasks)
        self.env.set_task(task)

        mt2 = metaworld.MT1(self.params['env_name_2'])
        self.env2 = mt2.train_classes[self.params['env_name_2']]()
        task = random.choice(mt2.train_tasks)
        self.env2.set_task(task)

        # import plotting (locally if 'obstacles' env)
        if not (self.params["env_name"] == "obstacles-cs285-v0"):
            import matplotlib

            matplotlib.use("Agg")

        # Maximum length for episodes
        self.params["ep_len"] = self.params["ep_len"] or self.env.max_path_length
        global MAX_VIDEO_LEN
        MAX_VIDEO_LEN = self.params["ep_len"]

        # Is this env continuous, or self.discrete?
        discrete = isinstance(self.env.action_space, gym.spaces.Discrete)
        # Are the observations images?
        img = len(self.env.observation_space.shape) > 2

        self.params["agent_params"]["discrete"] = discrete

        # Observation and action sizes

        ob_dim = (
            self.env.observation_space.shape
            if img
            else self.env.observation_space.shape[0]
        )
        ac_dim = self.env.action_space.n if discrete else self.env.action_space.shape[0]
        self.params["agent_params"]["ac_dim"] = ac_dim
        self.params["agent_params"]["ob_dim"] = ob_dim
        self.params["agent_params"]["period"] = self.params["period"]

        # simulation timestep, will be used for video saving
        if "model" in dir(self.env):
            self.fps = 1 / self.env.model.opt.timestep
        elif "env_wrappers" in self.params:
            self.fps = 30  # This is not actually used when using the Monitor wrapper
        elif "video.frames_per_second" in self.env.env.metadata.keys():
            self.fps = self.env.env.metadata["video.frames_per_second"]
        else:
            self.fps = 10

        #############
        ## AGENT
        #############

        agent_class = self.params["agent_class"]
        self.agent = agent_class(self.env, self.params["agent_params"])

    def run_training_loop(
        self,
        n_iter,
        collect_policy,
        eval_policy,
        initial_expertdata=None,
        relabel_with_expert=False,
        start_relabel_with_expert=1,
        expert_policy=None,
    ):
        """
        :param n_iter:  number of (dagger) iterations
        :param collect_policy:
        :param eval_policy:
        :param initial_expertdata:
        :param relabel_with_expert:  whether to perform dagger
        :param start_relabel_with_expert: iteration at which to start relabel with expert
        :param expert_policy:
        """


        # init vars at beginning of training
        self.total_envsteps = 0
        self.start_time = time.time()

        for itr in range(n_iter):
            print("\n\n********** Iteration %i ************" % itr)
            self.agent.update_time(0)

            # decide if videos should be rendered/logged at this iteration
            if (
                itr % self.params["video_log_freq"] == 0
                and self.params["video_log_freq"] != -1
            ):
                self.logvideo = True
            else:
                self.logvideo = False
            self.log_video = self.logvideo

            # decide if metrics should be logged
            if self.params["scalar_log_freq"] == -1:
                self.logmetrics = False
            elif itr % self.params["scalar_log_freq"] == 0:
                self.logmetrics = True
            else:
                self.logmetrics = False

            # collect trajectories, to be used for training
            training_returns = self.collect_training_trajectories(
                itr, initial_expertdata, collect_policy, self.params["batch_size"]
            )
            paths, envsteps_this_batch, train_video_paths = training_returns
            self.total_envsteps += envsteps_this_batch

            # add collected data to replay buffer
            self.agent.add_to_replay_buffer(paths)

            # train agent (using sampled data from replay buffer)
            self.agent.update_time(0)
            train_logs = self.train_agent()

            # log/save
            if self.logvideo or self.logmetrics:
                # perform logging
                print("\nBeginning logging procedure...")
                self.perform_logging(
                    itr, paths, eval_policy, train_video_paths, train_logs
                )

                if self.params["save_params"]:
                    self.agent.save(
                        "{}/agent_itr_{}.pt".format(self.params["logdir"], itr)
                    )

    def run_second_task_loop(
            self,
            n_iter,
            collect_policy,
            eval_policy,
            initial_expertdata=None,
            relabel_with_expert=False,
            start_relabel_with_expert=1,
            expert_policy=None,
    ):
        """
        :param n_iter:  number of (dagger) iterations
        :param collect_policy:
        :param eval_policy:
        :param initial_expertdata:
        :param relabel_with_expert:  whether to perform dagger
        :param start_relabel_with_expert: iteration at which to start relabel with expert
        :param expert_policy:
        """

        # init vars at beginning of training
        self.total_envsteps = 0
        self.start_time = time.time()
        self.agent.env = self.env2

        for itr in range(n_iter, 2*n_iter):
            self.second_task = True
            print("\n\n********** Iteration %i ************" % itr)

            # decide if videos should be rendered/logged at this iteration
            if (
                    itr % self.params["video_log_freq"] == 0
                    and self.params["video_log_freq"] != -1
            ):
                self.logvideo = True
            else:
                self.logvideo = False
            self.log_video = self.logvideo

            # decide if metrics should be logged
            if self.params["scalar_log_freq"] == -1:
                self.logmetrics = False
            elif itr % self.params["scalar_log_freq"] == 0:
                self.logmetrics = True
            else:
                self.logmetrics = False

            # collect trajectories, to be used for training
            training_returns = self.collect_training_trajectories(
                itr, initial_expertdata, collect_policy, self.params["batch_size"]
            )
            paths, envsteps_this_batch, train_video_paths = training_returns
            self.total_envsteps += envsteps_this_batch

            # add collected data to replay buffer
            self.agent.add_to_replay_buffer(paths)

            # train agent (using sampled data from replay buffer)
            self.agent.update_time(1)
            train_logs = self.train_agent()

            # log/save
            if self.logvideo or self.logmetrics:
                # perform logging
                self.agent.update_time(0)
                print("\nBeginning logging procedure...")
                self.perform_logging(
                    itr, paths, eval_policy, train_video_paths, train_logs
                )
                self.agent.update_time(1)
                if self.params["save_params"]:
                    self.agent.save(
                        "{}/agent_itr_{}.pt".format(self.params["logdir"], itr)
                    )

    ####################################
    ####################################

    def collect_training_trajectories(
        self, itr, load_initial_expertdata, collect_policy, batch_size,
    ):
        """
        :param itr:
        :param load_initial_expertdata:  path to expert data pkl file
        :param collect_policy:  the current policy using which we collect data
        :param batch_size:  the number of transitions we collect
        :return:
            paths: a list trajectories
            envsteps_this_batch: the sum over the numbers of environment steps in paths
            train_video_paths: paths which also contain videos for visualization purposes
        """
        # if your load_initial_expertdata is None, then you need to collect new trajectories at *every* iteration
        # TODO done decide whether to load training data or use
        # HINT: depending on if it's the first iteration or not,
        # decide whether to either
        # load the data. In this case you can directly return as follows
        # ``` return loaded_paths, 0, None ```

        # if it's the first iteration and you aren't loading data, then
        # `self.params['batch_size_initial']` is the number of transitions you want to collect
        if itr == 0 and load_initial_expertdata is not None:
            paths = pickle.load(open(self.params["expert_data"], "rb"))
            return paths, 0, None
        elif itr == 0 and load_initial_expertdata is None:
            num_transitions_to_sample = self.params["batch_size_initial"]
        else:
            num_transitions_to_sample = batch_size
            # collect data to be used for training
        print("\nCollecting data to be used for training...")
        num_transitions_to_sample = self.params["batch_size_initial"]
        if self.second_task:
            paths, envsteps_this_batch = utils.sample_trajectories(
                self.env2, collect_policy, num_transitions_to_sample, self.params["ep_len"]
            )
        else:
            paths, envsteps_this_batch = utils.sample_trajectories(
                self.env, collect_policy, num_transitions_to_sample, self.params["ep_len"]
            )

        # collect more rollouts with the same policy, to be saved as videos in tensorboard
        train_video_paths = None

        return paths, envsteps_this_batch, train_video_paths

    def train_agent(self):
        all_logs = []
        for train_step in range(self.params["num_agent_train_steps_per_iter"]):
            (
                ob_batch,
                ac_batch,
                re_batch,
                next_ob_batch,
                terminal_batch,
            ) = self.agent.sample(self.params["train_batch_size"])
            train_log = self.agent.train(
                ob_batch, ac_batch, re_batch, next_ob_batch, terminal_batch
            )
            all_logs.append(train_log)
        return all_logs

    ####################################
    ####################################

    def perform_logging(self, itr, paths, eval_policy, train_video_paths, all_logs):

        last_log = all_logs[-1]

        #######################

        # collect eval trajectories, for logging
        print("\nCollecting data for eval...")
        eval_paths, eval_envsteps_this_batch = utils.sample_trajectories(
            self.env, eval_policy, self.params["eval_batch_size"], self.params["ep_len"]
        )
        self.agent.update_time(1)
        eval_paths2, eval_envsteps_this_batch2 = utils.sample_trajectories(
            self.env2, eval_policy, self.params["eval_batch_size"], self.params["ep_len"]
        )

        #######################

        # save eval metrics
        if self.logmetrics:
            # returns, for logging
            train_returns = [path["reward"].sum() for path in paths]
            eval_returns = [eval_path["reward"].sum() for eval_path in eval_paths]
            eval_returns2 = [eval_path["reward"].sum() for eval_path in eval_paths2]

            # episode lengths, for logging
            train_ep_lens = [len(path["reward"]) for path in paths]
            eval_ep_lens = [len(eval_path["reward"]) for eval_path in eval_paths]
            eval_ep_lens2 = [len(eval_path["reward"]) for eval_path in eval_paths2]

            # decide what to log
            logs = OrderedDict()
            logs["Eval_AverageReturn"] = np.mean(eval_returns)
            logs["Eval_AverageReturn2"] = np.mean(eval_returns2)
            logs["Eval_StdReturn"] = np.std(eval_returns)
            logs["Eval_MaxReturn"] = np.max(eval_returns)
            logs["Eval_MinReturn"] = np.min(eval_returns)
            logs["Eval_AverageEpLen"] = np.mean(eval_ep_lens)
            logs["Eval_AverageEpLen2"] = np.mean(eval_ep_lens2)

            logs["Train_AverageReturn"] = np.mean(train_returns)
            logs["Train_StdReturn"] = np.std(train_returns)
            logs["Train_MaxReturn"] = np.max(train_returns)
            logs["Train_MinReturn"] = np.min(train_returns)
            logs["Train_AverageEpLen"] = np.mean(train_ep_lens)

            logs["Train_EnvstepsSoFar"] = self.total_envsteps
            logs["TimeSinceStart"] = time.time() - self.start_time
            logs.update(last_log)

            if itr == 0:
                self.initial_return = np.mean(train_returns)
            logs["Initial_DataCollection_AverageReturn"] = self.initial_return

            # perform the logging
            for key, value in logs.items():
                print("{} : {}".format(key, value))
                self.logger.log_scalar(value, key, itr)
            print("Done logging...\n\n")

            self.logger.flush()

