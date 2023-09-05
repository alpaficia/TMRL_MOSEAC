import numpy as np

def Action_adapter(a, max_action):
    # from [-1,1] to [-max,max]
    return a * max_action


def Action_t_relu6_adapter_reverse(act, max_action):
    # from [0, max] to [0,6]
    return (act * 6.0) / max_action


def Action_t_relu6_adapter(a, max_action):
    # from [0,6] to [0,max]
    return (a/6.0) * max_action


def Action_adapter_reverse(act, max_action):
    # from [-max,max] to [-1,1]
    return act / max_action

def reward_adapter(reward, alpha_t, energy_cost_per_step, alpha_epsilon, 
                   time, alpha_tau):
    reward_task = reward * alpha_t
    reward_energy = energy_cost_per_step * alpha_epsilon
    reward_time = time * alpha_tau
    return float(reward_task - reward_energy - reward_time)
