![Diagram](distributed_RL_diagram.jpg)

### A distributed system for Reinforcement Learning / Training

Support for multiple environments which produces a Trajectory into a replaybuffer. <br/>
The base model is a DQN. <br/>
Environments and Policy are stored in object storage , while the replaybuffer is on bigtable. <br/>
While DQN is an on-policy algorithm, in a distributed environment the collection policy is sometimes out of date compared to the training policy. In that case we are training off policy. <br/>

The code provided requires certain configuration / resources in order to work: <br/>
*We used Google Cloud Platform, but it may be possible to use a different service* <br/>
-A cloud service with: <br/>
  -object storage (GCP glob) <br/>
  -query based database (bigtable) <br/>
  -Docker orchestration (Kubernetes) <br/>
  -TPU / GPU allocation <br/>
-Service Authentication (eg; Application level credentials) <br/>
-An Environment which outputs (State Observations, Available Actions, previous_state_Reward) <br/>
-The code for our training and collections is opening source but the environment built on Unity is not. <br/>
  -We have included 2 open source environments /cartpole and /breakout to test. <br/>

##Deployment :
 With Docker Orchestration Deploy :
 https://github.com/jun-bun/rab-tf-agents/blob/master/deploy/Dockerfile
 *Note we use a specific docker build which supports Unity on Linux the base container is hosted here : https://hub.docker.com/r/tenserflow/gpu-unity-ubuntu-xfce-novnc*

![Demo](replay_demo.gif)

