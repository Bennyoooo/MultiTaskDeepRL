import metaworld
import random

mt1 = metaworld.MT1('push-v1') # Construct the benchmark, sampling tasks

env = mt1.train_classes['push-v1']()
task = random.choice(mt1.train_tasks)
env.set_task(task)
a = env.action_space.sample()
print(a)
