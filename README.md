# SRTP_Alpha_CC_Zero

- Alpha CC Zero: A Deep Reinforcement Learning Model for Chinese Chess
- Environment:

```
Python 3.6.13
cuda 11.2
tensorflow-gpu 2.6.2
pillow 8.4.0
scipy
```

- How to run:

```
python main.py --mode train --processor gpu --num_gpus 1 --res_block_nums 7 --train_epoch 100
python main.py --mode play --processor gpu --num_gpus 1 --ai_function mcts --ai_count 1 
```
