<<<<<<< HEAD
# Deep Dyna-Q: Integrating Planning for Task-Completion Dialogue Policy Learning
*An implementation of the  
[Deep Dyna-Q: Integrating Planning for Task-Completion Dialogue Policy Learning](https://arxiv.org/abs/1801.06176)*

This document describes how to run the simulation of DDQ Agent.

## Content
* [Data](#data)
* [Parameter](#parameter)
* [Running Dialogue Agents](#running-dialogue-agents)
* [Evaluation](#evaluation)
* [Reference](#reference)

## Data
all the data is under this folder: ./src/deep_dialog/data

* Movie Knowledge Bases<br/>
`movie_kb.1k.p` --- 94% success rate (for `user_goals_first_turn_template_subsets.v1.p`)<br/>
`movie_kb.v2.p` --- 36% success rate (for `user_goals_first_turn_template_subsets.v1.p`)

* User Goals<br/>
`user_goals_first_turn_template.v2.p` --- user goals extracted from the first user turn<br/>
`user_goals_first_turn_template.part.movie.v1.p` --- a subset of user goals [Please use this one, the upper bound success rate on movie_kb.1k.json is 0.9765.]

* NLG Rule Template<br/>
`dia_act_nl_pairs.v6.json` --- some predefined NLG rule templates for both User simulator and Agent.

* Dialog Act Intent<br/>
`dia_acts.txt`

* Dialog Act Slot<br/>
`slot_set.txt`

## Parameter

### Basic setting

`--agt`: the agent id<br/>
`--usr`: the user (simulator) id<br/>
`--max_turn`: maximum turns<br/>
`--episodes`: how many dialogues to run<br/>
`--slot_err_prob`: slot level err probability<br/>
`--slot_err_mode`: which kind of slot err mode<br/>
`--intent_err_prob`: intent level err probability

### DDQ Agent setting
`--grounded`: planning k steps with environment rather than world model, serving as a upper bound.<br/>
`--boosted`: boost the world model with examles generated by rule agent<br/>
`--train_world_model`: train world model on the fly<br/>


### Data setting

`--movie_kb_path`: the movie kb path for agent side<br/>
`--goal_file_path`: the user goal file path for user simulator side

### Model setting

`--dqn_hidden_size`: hidden size for RL agent<br/>
`--batch_size`: batch size for DDQ training<br/>
`--simulation_epoch_size`: how many dialogue to be simulated in one epoch<br/>
`--warm_start`: use rule policy to fill the experience replay buffer at the beginning<br/>
`--warm_start_epochs`: how many dialogues to run in the warm start

### Display setting

`--run_mode`: 0 for display mode (NL); 1 for debug mode (Dia_Act); 2 for debug mode (Dia_Act and NL); >3 for no display (i.e. training)<br/>
`--act_level`: 0 for user simulator is Dia_Act level; 1 for user simulator is NL level<br/>
`--auto_suggest`: 0 for no auto_suggest; 1 for auto_suggest<br/>
`--cmd_input_mode`: 0 for NL input; 1 for Dia_Act input. (this parameter is for AgentCmd only)

### Others

`--write_model_dir`: the directory to write the models<br/>
`--trained_model_path`: the path of the trained RL agent model; load the trained model for prediction purpose.

`--learning_phase`: train/test/all, default is all. You can split the user goal set into train and test set, or do not split (all); We introduce some randomness at the first sampled user action, even for the same user goal, the generated dialogue might be different.<br/>

## Running Dialogue Agents

Train DDQ Agent with K planning steps:
```sh
python run.py --agt 9 --usr 1 --max_turn 40 
	      --movie_kb_path ./deep_dialog/data/movie_kb.1k.p 
	      --dqn_hidden_size 80 --experience_replay_pool_size 5000 
	      --episodes 500 
	      --simulation_epoch_size 100 
	      --run_mode 3 
	      --act_level 0 
	      --slot_err_prob 0.0 
	      --intent_err_prob 0.00 
	      --batch_size 16 
	      --goal_file_path ./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p 
	      --warm_start 1 --warm_start_epochs 100 
	      --planning_steps K-1 
	      --write_model_dir ./deep_dialog/checkpoints/DDQAgent
	      --torch_seed 100
	      --grounded 0
	      --boosted 1
	      --train_world_model 1

```
Test RL Agent with N dialogues:
```sh
python run.py --agt 9 --usr 1 --max_turn 40
	      --movie_kb_path ./deep_dialog/data/movie_kb.1k.p
	      --dqn_hidden_size 80
	      --experience_replay_pool_size 1000
	      --episodes 300 
	      --simulation_epoch_size 100
	      --write_model_dir ./deep_dialog/checkpoints/DDQAgent/
	      --slot_err_prob 0.00
	      --intent_err_prob 0.00
	      --batch_size 16
	      --goal_file_path ./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p
	      --trained_model_path ./deep_dialog/checkpoints/DDQAgent/TRAINED_MODEL
	      --run_mode 3
```
## Experiments
To run the scripts, move the two bash files under src folder. 
1. Bash_figure_4.sh is the script for figure 4.
2. Bash_figure_5.sh is the script for figure 5. 

## Evaluation
To evaluate the performance of agents, three metrics are available: success rate, average reward, average turns. Here we show the learning curve with success rate.

1. Plotting Learning Curve
``` python draw_learning_curve.py --result_file ./deep_dialog/checkpoints/DDQAgent/noe2e/TRAINED_MODEL.json```
2. Pull out the numbers and draw the curves in Excel

## Reference

Main papers to be cited
```
@inproceedings{Peng2018DeepDynaQ,
  title={Deep Dyna-Q: Integrating Planning for Task-Completion Dialogue Policy Learning},
  author={Peng, Baolin and Li, Xiujun and Gao, Jianfeng and Liu, Jingjing and Wong, Kam-Fai and Su, Shang-Yu},
  booktitle={ACL},
  year={2018}
}

@article{li2016user,
  title={A User Simulator for Task-Completion Dialogues},
  author={Li, Xiujun and Lipton, Zachary C and Dhingra, Bhuwan and Li, Lihong and Gao, Jianfeng and Chen, Yun-Nung},
  journal={arXiv preprint arXiv:1612.05688},
  year={2016}
}
=======
# DDQ_erl
>>>>>>> 63ba092abeb0fea09ae2a58c4551343f9ae99a6d
