![Diagram](distributed_RL_diagram.jpg)

### A distributed system for Reinforcement Learning / Training

Support for multiple environments which produces a Trajectory into a replaybuffer. <br/>
The base model is a DQN. <br/>
Environments and Policy are stored in object storage , while the replaybuffer is on bigtable. <br/>
While DQN is an on-policy algorithm, in a distributed environment the collection policy is sometimes stale compared to the training policy. In this case, we are training off policy. <br/>

The code provided requires certain configuration / resources in order to run: <br/>
*We used Google Cloud Platform, but it may be possible to use other services* <br/>
-A cloud service with: <br/>
  -object storage (GCP glob) <br/>
  -query based database (bigtable) <br/>
  -Docker orchestration (Kubernetes) <br/>
  -TPU / GPU allocation <br/>
-Service Authentication <br/>
-An Environment which outputs (State Observations, Available Actions, previous_state_Reward) <br/>
  -We have included 2 open source environments /cartpole and /breakout to test. <br/>
  -/crane contains the ML training code but lacks the environment because it is currently closed source. 

## Multi-Environment Deployment : <br/>
 With Docker Orchestration Deploy : <br/>
 https://github.com/jun-bun/rab-tf-agents/blob/master/deploy/Dockerfile <br/>
 *Note we use a specific docker build which supports Unity on Linux. The base container is hosted here : https://hub.docker.com/r/tenserflow/gpu-unity-ubuntu-xfce-novnc* <br/>

## Single Environment Test : <br/>
  python3 -m breakout.collect_to_bigtable<br/>

## Training: <br/>
  python3 -m breakout.train_from_bigtable <br>

![Demo](replay_demo.gif)

