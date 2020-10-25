Local Conda environment setup

```
conda create -n drl python=3.6
source activate drl
export LD_LIBRARY_PATH=~/.mujoco/mujoco200/bin/
pip install --user -r requirements.txt
```

