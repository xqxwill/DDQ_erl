#Below is the script used for figure 4
for ((i=1; i<= 5; i++));do
let "seed=$i*100"
python run_ppo.py --agt 11 \
--usr 1 --max_turn 40 --movie_kb_path ./deep_dialog/data/movie_kb.1k.p --dqn_hidden_size 80 \
--experience_replay_pool_size 5000 --episodes 500 --simulation_epoch_size 100 \
--run_mode 3 --act_level 0 --slot_err_prob 0.0 --intent_err_prob 0.00 --batch_size 16 \
--goal_file_path ./deep_dialog/data/user_goals_first_turn_template.part.movie.v1.p \
--warm_start 1 --warm_start_epochs 100 \
--write_model_dir ./deep_dialog/checkpoints/PPO_$i \
--planning_steps 0 --torch_seed $seed --grounded 0 --boosted 0 --train_world_model 0
done
