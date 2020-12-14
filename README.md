Local Conda environment setup

```
conda create -n drl python=3.6
source activate drl
export LD_LIBRARY_PATH=~/.mujoco/mujoco200/bin/
pip install --user -r requirements.txt
```

Experiment commands:
```
## larger network
python scripts/run_hw2.py --two_tasks --exp_name debug --env_name box-close-v1 --env_name_2 button-press-v1 -b 3000 -n 150 --period 2 -l 4
python scripts/run_hw2.py --two_tasks --exp_name debug --env_name box-close-v1 --env_name_2 button-press-v1 -b 3000 -n 150 --period 2 -l 8
python scripts/run_hw2.py --two_tasks --exp_name debug --env_name box-close-v1 --env_name_2 button-press-v1 -b 3000 -n 150 --period 2 -s 128 
python scripts/run_hw2.py --two_tasks --exp_name debug --env_name box-close-v1 --env_name_2 button-press-v1 -b 3000 -n 150 --period 2 -s 256
python scripts/run_hw2.py --two_tasks --exp_name debug --env_name box-close-v1 --env_name_2 button-press-v1 -b 3000 -n 150 --period 2 -l 4 -s 128
python scripts/run_hw2.py --two_tasks --exp_name debug --env_name box-close-v1 --env_name_2 button-press-v1 -b 3000 -n 150 --period 2 -l 8 -s 256

## learning rate
python scripts/run_hw2.py --two_tasks --exp_name debug --env_name box-close-v1 --env_name_2 button-press-v1 -b 3000 -n 150 --period 2 -lr 0.001
python scripts/run_hw2.py --two_tasks --exp_name debug --env_name box-close-v1 --env_name_2 button-press-v1 -b 3000 -n 150 --period 2 -lr 0.01
python scripts/run_hw2.py --two_tasks --exp_name debug --env_name box-close-v1 --env_name_2 button-press-v1 -b 3000 -n 150 --period 2 -lr 0.05

## batch size
python scripts/run_hw2.py --two_tasks --exp_name debug --env_name box-close-v1 --env_name_2 button-press-v1 -b 5000 -n 150 --period 2
python scripts/run_hw2.py --two_tasks --exp_name debug --env_name box-close-v1 --env_name_2 button-press-v1 -b 7000 -n 150 --period 2
python scripts/run_hw2.py --two_tasks --exp_name debug --env_name box-close-v1 --env_name_2 button-press-v1 -b 1000 -n 150 --period 2

```

