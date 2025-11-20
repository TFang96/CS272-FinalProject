import gymnasium
import highway_env
from matplotlib import pyplot as plt

def create_highway_env():
    env = gymnasium.make('highway-v0', render_mode='rgb_array')
    env.reset()
    return env

def create_merge_env():
    env = gymnasium.make('merge-v0', render_mode='rgb_array')
    env.reset()
    return env

def create_intersection_env():
    env = gymnasium.make('intersection-v1', render_mode='rgb_array')
    env.reset()
    return env
'''
def main():
    plt.imshow(create_highway_env().render())
    plt.show()
    plt.imshow(create_highway_env().render())
    plt.show()
    plt.imshow(create_intersection_env().render())
    plt.show()

main()
'''