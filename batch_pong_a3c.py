import gym
import numpy as np
import tensorflow as tf
from misc import *
import time
import ray
# from PGA_online import PolicyGradientAgent
from actorcritic import ActorCritic
from atari_environment import AtariEnvironment
from conv_a3c import ConvAC

NUM_WORKERS = 1
GAMMA = 0.99
GYM_ENV = 'Pong-v0'
env_batch = 5

ray.init(num_workers=NUM_WORKERS)

def env_init():
    return [AtariEnvironment(gym.make(GYM_ENV)) for i in range(env_batch)]

def env_reinit(env_list):
    return env_list

ray.env.env_list = ray.EnvironmentVariable(env_init, env_reinit)

def ac_init():
    env_list = ray.env.env_list
    hparams = {
            'input_size': list(env.state_size),
            # 'hidden': 64,
            'num_actions': env.action_space_size,
            'learning_rate': 0.001,
            'entropy_wt': 0.01
    }
    return ConvAC(hparams)

def ac_reinit(actor_critic):
    return actor_critic

ray.env.actor_critic = ray.EnvironmentVariable(ac_init, ac_reinit)

def rollout_process(env, actor_critic, steps):
    obs, acts, rews, done = policy_continue(env, actor_critic, steps)
    estimated_values = actor_critic.get_value(obs).flatten()
    assert len(estimated_values.shape) == 1
    cur_rwd = 0 if done else estimated_values[-1]
    rewards = []
    for r in reversed(rews):
        cur_rwd = r + GAMMA * cur_rwd 
        rewards.insert(0, cur_rwd)
    rewards = np.asarray(rewards)
    norm_advs = normalize(rewards - estimated_values)
    return {"obs": obs, 
            "acts": acts, 
            "norm_advs": norm_advs, 
            "rewards": rewards,
            "done": done}

@ray.remote
def a3c_rollout_grad(params, avg_rwd):
    prelim_start = time.time()
    GAMMA = params['gamma']
    env_list = ray.env.env_list
    actor_critic = ray.env.actor_critic
    steps = 10

    actor_critic.set_weights(params['weights'])
    obs, acts, norm_advs, rewards = [], [], [], []
    done = 0

    for env in env_list:
        results = rollout_process(env, actor_critic)
        obs.extend(results["obs"])
        acts.extend(results["acts"])
        norm_advs.extend(results["norm_advs"])
        rewards.extend(results["rewards"])
        done += results["done"]
    # print "Bootstrapped Obs Rewards: ", rewards
    # print "Estimated Values: ", estimated_values
    grads = actor_critic.compute_gradients(obs, acts, norm_advs, rewards)

    info = {"done": done, 
            "obs": obs, 
            "rews": rewards, 
            "real_rwds": rews}
    # if any([(np.isnan(i)).any() for i in grads]):
    #     import ipdb; ipdb.set_trace()  # breakpoint 3e26a68a //


    return grads, info

def train(u_itr=5000):
    actor_critic = ray.env.actor_critic
    remaining = []
    critic_loss = []
    rwds = []
    obs = []
    cur_itr = 0
    tasks_launched = 0
    params = {"gamma": GAMMA}
    average_reward = RunningAvg()    

    ## debug
    sq_loss = lambda x, y: sum((x - y)**2)
    cur_time = time.time()
    timing = {"setuptime": 0, 
                "policytime": 0, 
                "gradtime": 0,
                "starttask": 0,
                "updatetime": 0,
                "launch": 0,
                "get": 0,
                "st_log": []
                }

    while cur_itr < u_itr:
        start_task = time.time()
        params['weights'] = actor_critic.get_weights()        
        # if any([(np.isnan(i)).any() for i in params['weights'].values()]):
        #     import ipdb; ipdb.set_trace()  # breakpoint 71025765 //
        
        param_id = ray.put(params)

        jobs = NUM_WORKERS - len(remaining)
        remaining.extend([a3c_rollout_grad.remote(params, average_reward.val()) for i in range(jobs)])
        run_task = time.time()
        result, remaining = ray.wait(remaining)
        get_task = time.time()
        grads, info = ray.get(result)[0]
        average_reward.add(info["rews"])
        rwds.extend(info["real_rwds"])
        obs.extend(info["obs"])

        actor_critic.model_update(grads)
        update_task = time.time()


        cur_itr += int(info["done"])
        timing["policytime"] += info["policytime"]
        timing["gradtime"] += info["gradtime"]
        timing["setuptime"] += info["setuptime"]
        timing["starttask"] += run_task - start_task
        timing["st_log"].append(int((run_task - start_task) * 1000))
        timing["updatetime"] += update_task - get_task
        timing["launch"] += info["start"] - run_task
        timing["get"] += get_task - info["end"]
        tasks_launched += 1        

        if cur_itr and cur_itr % 50 == 0:
            testbed = AtariEnvironment(gym.make(GYM_ENV))
            print "%d: Avg Reward - %f" % (cur_itr, evaluate_policy(testbed, actor_critic, itr=10))
            c_val = np.asarray(discounted_cumsum(rwds, GAMMA))
            c_est = actor_critic.get_value(obs).flatten()
            print "%d: Critic Loss - total: %f \t avg: %f \t most: %f" % (cur_itr, 
                sq_loss(c_val, c_est), 
                sq_loss(c_val, c_est) / len(c_val), 
                sq_loss(c_val[:-10], c_est[:-10]) / (len(c_val) - 10 + 1e-2))
            cur_itr += 1
        if info["done"]:
            # print "That took {} seconds..".format(time.time() - cur_time)
            # print "Launched {} tasks...".format(tasks_launched)
            # print "Start Time: {}".format(timing['starttask'])
            # print "In-Worker Setup Time: {}".format(timing['setuptime'])
            # print "Policy Time: {}".format(timing['policytime'])
            # print "Gradient Time: {}".format(timing['gradtime'])
            # print "Update Time: {}".format(timing['updatetime'])
            # print "Task Launch Time: {}".format(timing['launch'])
            # print "Task Get Time: {}".format(timing['get'])
            # print "Log - ", timing["st_log"]

            cur_time = time.time()
            tasks_launched = 0
            timing = {"setuptime": 0, 
                        "policytime": 0, 
                        "gradtime": 0,
                        "starttask": 0,
                        "updatetime": 0,
                        "launch": 0,
                        "get": 0,
                        "st_log": []
                        }

            rwds = []
            obs = []

if __name__ == '__main__':
    train()
