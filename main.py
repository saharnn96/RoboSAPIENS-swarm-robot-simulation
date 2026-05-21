import irsim

import irsim

env = irsim.make('robot_world.yaml')
env.load_behavior("custom_behavior_methods")

for i in range(1500):

    env.step()
    env.render(0.05)
    if env.done():
        break

env.end(3)

