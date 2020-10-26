import metaworld
import random

mt1 = metaworld.MT1('pick-place-v1') # Construct the benchmark, sampling tasks

env = mt1.train_classes['pick-place-v1']()
task = random.choice(mt1.train_tasks)
env.set_task(task)
