import numpy as np
from attackgraph import file_op as fp
from baselines.common import models
import os
from baselines.deepq.deepq import learn_multi_nets, learn


DIR = os.getcwd() + '/'
DIR_def = os.getcwd() + '/defender_strategies/'
DIR_att = os.getcwd() + '/attacker_strategies/'
# print(DIR_att)


#TODO: str in str_set should include .pkl
def sample_strategy_from_mixed(env, str_set, mix_str, identity):

    if not isinstance(mix_str,np.ndarray):
        raise ValueError("mix_str in sample func is not a numpy array.")

    if not len(str_set) == len(mix_str):
        raise ValueError("Length of mixed strategies does not match number of strategies.")

    picked_str = np.random.choice(str_set, p=mix_str)
    if not fp.isInName('.pkl', name = picked_str):
        raise ValueError('The strategy picked is not a pickle file.')

    # TODO: Transfer learning modification
    if identity == 0: # pick a defender's strategy
        path = DIR + 'defender_strategies/'
        scope = "def_str_epoch" + str(0) + '.pkl'
    elif identity == 1:
        path = DIR + 'attacker_strategies/'
        scope = 'att_str_epoch' + str(1) + '.pkl'
    else:
        raise ValueError("identity is neither 0 or 1!")

    # print(path + picked_str)
    if not fp.isExist(path + picked_str):
        raise ValueError('The strategy picked does not exist!')

    if "epoch1" in picked_str:
        act = fp.load_pkl(path + picked_str)
        return act

    flag = env.training_flag
    env.set_training_flag(identity)

    act = learn(
        env,
        network=models.mlp(num_hidden=256, num_layers=1), #TODO: hard coding.
        total_timesteps=0,
        load_path= path + picked_str,
        scope = scope + '/'  #picked_str + '/'
    )

    env.set_training_flag(flag)

    return act













