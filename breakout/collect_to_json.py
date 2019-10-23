import os
import argparse
import datetime
from tqdm import tqdm

import numpy as np
import tensorflow as tf

from google.oauth2 import service_account

from protobuf.experience_replay_pb2 import Trajectory, Info
from breakout.dqn_model import DQN_Model
from util.gcp_io import gcp_load_pipeline, gcs_load_weights, gcs_save_weights, cbt_global_iterator, cbt_read_rows
from util.logging import TimeLogger

import json
import collections
import gin

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
#np.set_printoptions(threshold=np.inf)

""" Collect Breakout TF-agents DQN

"""

SCOPES = ['https://www.googleapis.com/auth/cloud-platform']
SERVICE_ACCOUNT_FILE = 'cbt_credentials.json'

#SET HYPERPARAMETERS
VECTOR_OBS_SPEC = [4]
VISUAL_OBS_SPEC = [210,160,3]
NUM_ACTIONS=2
CONV_LAYER_PARAMS=((8,4,32),(4,2,64),(3,1,64))
FC_LAYER_PARAMS=(512,200)
LEARNING_RATE=0.00042
GAMMA = 0.9

@gin.configurable
def train_eval(
    root_dir,
    env_name='Breakout-v0',
    num_iterations=100000,
    train_sequence_length=1,
    # Params for QNetwork
    fc_layer_params=(500,),
    # Params for QRnnNetwork
    input_fc_layer_params=(50,),
    lstm_size=(20,),
    output_fc_layer_params=(20,),

    # Params for collect
    initial_collect_steps=1000,
    collect_steps_per_iteration=1,
    epsilon_greedy=0.1,
    ##replay_buffer_capacity=100000,
    ):

    print("-> Initializing Gym environement...")
    env = tf_py_environment.TFPyEnvironment(suite_gym.load('Breakout-v0'))
    print("-> Environment intialized.")

    #Initialize Q_Network
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

    collect_driver = dynamic_step_driver.DynamicStepDriver(
        tf_env,
        collect_policy,
        observers=[replay_buffer.add_batch] + train_metrics,
        num_steps=collect_steps_per_iteration)

if __name__ == '__main__':
    #COMMAND-LINE ARGUMENTS
    parser = argparse.ArgumentParser('Environment-To-Bigtable Script')
    parser.add_argument('--gcp-project-id', type=str, default='for-robolab-cbai')
    parser.add_argument('--cbt-instance-id', type=str, default='rab-rl-bigtable')
    parser.add_argument('--cbt-table-name', type=str, default='breakout-experience-replay')
    parser.add_argument('--bucket-id', type=str, default='rab-rl-bucket')
    parser.add_argument('--prefix', type=str, default='breakout')
    parser.add_argument('--tmp-weights-filepath', type=str, default='/tmp/model_weights_tmp.h5')
    parser.add_argument('--train-epochs', type=int, default=1000000)
    parser.add_argument('--train-steps', type=int, default=10)
    parser.add_argument('--period', type=int, default=10)
    parser.add_argument('--output-dir', type=str, default='/tmp/training/')
    parser.add_argument('--log-time', default=False, action='store_true')
    args = parser.parse_args()

    #INSTANTIATE CBT TABLE AND GCS BUCKET
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
    cbt_table, gcs_bucket = gcp_load_pipeline(args.gcp_project_id, args.cbt_instance_id, args.cbt_table_name, args.bucket_id, credentials)

    print("Done Collecting --- Reading file")
    with open(r'collection.json', 'r') as file_reader :
        json_data = [json.loads(line) for line in file_reader]
        #print (json_data)
        count = 0
        for i in json_data:
            print ("count: ", count)
            count += 1
        print (json_data)
    print("-> Done!")