import os
import argparse
import datetime
import time
import struct
from tqdm import tqdm

import numpy as np
import tensorflow as tf

from google.oauth2 import service_account

from protobuf.experience_replay_pb2 import Trajectory, Info
from util.gcp_io import gcp_load_pipeline, gcs_load_weights, cbt_global_iterator
from util.logging import TimeLogger

import gym

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common

tf.compat.v1.enable_v2_behavior()

SCOPES = ['https://www.googleapis.com/auth/cloud-platform']
SERVICE_ACCOUNT_FILE = 'cbt_credentials.json'

#SET HYPERPARAMETERS
VECTOR_OBS_SPEC = [4]
VISUAL_OBS_SPEC = [210,160,3]
NUM_ACTIONS=2
CONV_LAYER_PARAMS=((8,4,32),(4,2,64),(3,1,64))
#FC_LAYER_PARAMS=(512,200)
FC_LAYER_PARAMS=(512,)
LEARNING_RATE=0.00042
EPSILON = 0.5

num_iterations = 20000 # @param {type:"integer"}

initial_collect_steps = 1000  # @param {type:"integer"} 
collect_steps_per_iteration = 1  # @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}


if __name__ == '__main__':
    #COMMAND-LINE ARGUMENTS
    parser = argparse.ArgumentParser('Environment-To-Bigtable Script')
    parser.add_argument('--gcp-project-id', type=str, default='for-robolab-cbai')
    parser.add_argument('--cbt-instance-id', type=str, default='rab-rl-bigtable')
    parser.add_argument('--cbt-table-name', type=str, default='breakout-experience-replay')
    parser.add_argument('--bucket-id', type=str, default='rab-rl-bucket')
    parser.add_argument('--prefix', type=str, default='breakout')
    parser.add_argument('--tmp-weights-filepath', type=str, default='/tmp/model_weights_tmp.h5')
    parser.add_argument('--num-cycles', type=int, default=1000000)
    parser.add_argument('--num-episodes', type=int, default=10)
    parser.add_argument('--max-steps', type=int, default=10)
    parser.add_argument('--log-time', default=False, action='store_true')
    args = parser.parse_args()

    #INSTANTIATE CBT TABLE AND GCS BUCKET
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    cbt_table, gcs_bucket = gcp_load_pipeline(args.gcp_project_id, args.cbt_instance_id, args.cbt_table_name, args.bucket_id, credentials)
    cbt_batcher = cbt_table.mutations_batcher(flush_count=args.num_episodes, max_row_bytes=500000000)
                                                                                           #104857600
    #INITIALIZE ENVIRONMENT
    print("-> Initializing Gym environement...")
    #env = gym.make('Breakout-v0')
    env = tf_py_environment.TFPyEnvironment(suite_gym.load('Breakout-v0'))
    print("-> Environment intialized.")

    q_net = q_network.QNetwork(
    env.observation_spec(),
    env.action_spec(),
    fc_layer_params=FC_LAYER_PARAMS)

    """ Now use tf_agents.agents.dqn.dqn_agent to instantiate a DqnAgent. In addition to the time_step_spec, 
        action_spec and the QNetwork, the agent constructor also requires an optimizer (in this case, AdamOptimizer), 
        a loss function, and an integer step counter.
    """
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    train_step_counter = tf.Variable(0)
    agent = dqn_agent.DqnAgent(
        env.time_step_spec(),
        env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter)

    agent.initialize()

    collect_policy = agent.collect_policy

    random_policy = random_tf_policy.RandomTFPolicy(env.time_step_spec(),
                                                env.action_spec())
    #Data Collection
    #@test {"skip": true}
    def collect_step(environment, policy):
        time_step = environment.current_time_step()
        action_step = policy.action(time_step)
        next_time_step = environment.step(action_step.action)
        
        print("timestep:")
        print(time_step)
        print("time step type:", type(time_step))
        print("actionstep")
        print(action_step)
        print("action_step type", type(action_step))
        print("next_time_step")
        print(next_time_step)
        print("next_time_step type", type(next_time_step))

        return time_step, action_step, next_time_step
        #Train code
        #traj = trajectory.from_transition(time_step, action_step, next_time_step)

        # Add trajectory to the replay buffer
        # Train code
        # buffer.add_batch(traj)

    def collect_data(env, policy, steps):
        for _ in range(steps):
            time_step, action_step, next_time_step = collect_step(env, policy)

    # This loop is so common in RL, that we provide standard implementations. 
    # For more details see the drivers module.
    # https://github.com/tensorflow/agents/blob/master/tf_agents/docs/python/tf_agents/drivers.md

    #LOAD MODEL
    #model = DQN_Model(input_shape=env.observation_space.shape,
    #                 num_actions=env.action_space.n,
    #                 conv_layer_params=CONV_LAYER_PARAMS,
    #                 fc_layer_params=FC_LAYER_PARAMS,
    #                 learning_rate=LEARNING_RATE)

    #GLOBAL ITERATOR
    global_i = cbt_global_iterator(cbt_table)
    print("global_i = {}".format(global_i))

    if args.log_time is True:
        time_logger = TimeLogger(["Collect Data" , "Serialize Data", "Write Cells", "Mutate Rows"], num_cycles=args.num_episodes)

    #COLLECT DATA FOR CBT
    print("-> Starting data collection...")
    rows = []
    for cycle in range(args.num_cycles):
        #gcs_load_weights(model, gcs_bucket, args.prefix, args.tmp_weights_filepath)
        for i in tqdm(range(args.num_episodes), "Cycle {}".format(cycle)):
            if args.log_time is True: time_logger.reset()

            #RL LOOP GENERATES A TRAJECTORY

            collect_data(env, random_policy, steps=args.num_episodes)

        """
            #observations, actions, rewards = [], [], []
            #obs = np.asarray(env.reset() / 255).astype(float)
            reward = 0
            done = False
            
            for _ in range(args.max_steps):
                action = model.step_epsilon_greedy(obs, EPSILON)
                new_obs, reward, done, info = env.step(action)

                observations.append(obs)
                actions.append(action)
                rewards.append(reward)

                if done: break
                obs = np.asarray(new_obs / 255).astype(float)
            
            if args.log_time is True: time_logger.log(0)

            #BUILD PB2 OBJECTS
            traj, info = Trajectory(), Info()
            traj.visual_obs.extend(np.asarray(observations).flatten())
            traj.actions.extend(actions)
            traj.rewards.extend(rewards)
            info.visual_obs_spec.extend(VISUAL_OBS_SPEC)
            info.num_steps = len(actions)

            if args.log_time is True: time_logger.log(1)

            #WRITE TO AND APPEND ROW
            row_key_i = i + global_i + (cycle * args.num_episodes)
            row_key = '{}_trajectory_{}'.format(args.prefix,row_key_i).encode()
            row = cbt_table.row(row_key)
            row.set_cell(column_family_id='trajectory',
                        column='traj'.encode(),
                        value=traj.SerializeToString())
            row.set_cell(column_family_id='trajectory',
                        column='info'.encode(),
                        value=info.SerializeToString())
            rows.append(row)
            
            if args.log_time is True: time_logger.log(2)
        
        gi_row = cbt_table.row('global_iterator'.encode())
        gi_row.set_cell(column_family_id='global',
                        column='i'.encode(),
                        value=struct.pack('i',row_key_i+1),
                        timestamp=datetime.datetime.utcnow())
        rows.append(gi_row)
        cbt_batcher.mutate_rows(rows)
        cbt_batcher.flush()

        if args.log_time is True: time_logger.log(3)
        if args.log_time is True: time_logger.print_logs()
        
        rows = []
        print("-> Saved trajectories {} - {}.".format(row_key_i - (args.num_episodes-1), row_key_i))
        """
    env.close()
    print("-> Done!")