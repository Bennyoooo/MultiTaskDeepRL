## larger network
python3 scripts/run_hw2.py --two_tasks --exp_name layer4 --env_name box-close-v1 --env_name_2 button-press-v1 -b 3000 -n 150 --period 2 -l 4
python3 scripts/run_hw2.py --two_tasks --exp_name layer8 --env_name box-close-v1 --env_name_2 button-press-v1 -b 3000 -n 150 --period 2 -l 8
python3 scripts/run_hw2.py --two_tasks --exp_name size128 --env_name box-close-v1 --env_name_2 button-press-v1 -b 3000 -n 150 --period 2 -s 128 
python3 scripts/run_hw2.py --two_tasks --exp_name size256 --env_name box-close-v1 --env_name_2 button-press-v1 -b 3000 -n 150 --period 2 -s 256
python3 scripts/run_hw2.py --two_tasks --exp_name layer4size128 --env_name box-close-v1 --env_name_2 button-press-v1 -b 3000 -n 150 --period 2 -l 4 -s 128
python3 scripts/run_hw2.py --two_tasks --exp_name layer8size256 --env_name box-close-v1 --env_name_2 button-press-v1 -b 3000 -n 150 --period 2 -l 8 -s 256

## learning rate
python3 scripts/run_hw2.py --two_tasks --exp_name lr1e-3 --env_name box-close-v1 --env_name_2 button-press-v1 -b 3000 -n 150 --period 2 -lr 0.001
python3 scripts/run_hw2.py --two_tasks --exp_name lr1e-2 --env_name box-close-v1 --env_name_2 button-press-v1 -b 3000 -n 150 --period 2 -lr 0.01
python3 scripts/run_hw2.py --two_tasks --exp_name lr5e-2 --env_name box-close-v1 --env_name_2 button-press-v1 -b 3000 -n 150 --period 2 -lr 0.05

## batch size
python3 scripts/run_hw2.py --two_tasks --exp_name batch5000 --env_name box-close-v1 --env_name_2 button-press-v1 -b 5000 -n 150 --period 2
python3 scripts/run_hw2.py --two_tasks --exp_name batch7000 --env_name box-close-v1 --env_name_2 button-press-v1 -b 7000 -n 150 --period 2
python3 scripts/run_hw2.py --two_tasks --exp_name batch1000 --env_name box-close-v1 --env_name_2 button-press-v1 -b 1000 -n 150 --period 2