import os
from attackgraph import json_op as jp

def nn_param():
    param = {}
    param['num_hidden'] = 256
    param['num_layers'] = 1
    param['lr'] = 5e-5
    param['total_timesteps_att'] = 700000
    param['total_timesteps_def'] = 1000000
    param['exploration_fraction'] = 0.5
    param['exploration_final_eps'] = 0.03
    param['print_freq'] = 250
    param['param_noise'] = False
    param['gamma'] = 0.99
    param['prioritized_replay'] = False
    param['checkpoint_freq'] = 30000

    #hado
    param['retrain_timesteps'] = 400000
    param['hado_param'] = (4, 0.7, 0.286)
    param['retrain_freq'] = 100000

    #simulation
    param['num_episodes'] = 250
    param['threshold'] = 0.1

    # transfer learning
    param['trans_timesteps_att'] = 700000
    param['trans_timesteps_def'] = 1000000
    param['trans_lr'] = 8e-5
    param['trans_exploration_fraction'] = 0.3
    param['trans_exploration_final_eps'] = 0.03

    #TODO: defender and attacker should have different param.
    param['trans_exploration_fraction_att'] = 0.5
    param['trans_exploration_final_eps_att'] = 0.03
    param['trans_exploration_fraction_def'] = 0.5
    param['trans_exploration_final_eps_def'] = 0.03

    param_path = os.getcwd() + '/network_parameters/param.json'
    jp.save_json_data(param_path, param)
    print("Network parameters have been saved in a json file successfully.")




def nn_param1():
    param = {}
    param['num_hidden'] = 256
    param['num_layers'] = 1
    param['lr'] = 5e-5
    param['total_timesteps_att'] = 3000
    param['total_timesteps_def'] = 3000
    param['exploration_fraction'] = 0.5
    param['exploration_final_eps'] = 0.03
    param['print_freq'] = 250
    param['param_noise'] = False
    param['gamma'] = 0.99
    param['prioritized_replay'] = True
    param['checkpoint_freq'] = None

    #hado
    param['retrain_timesteps'] = 2000
    param['hado_param'] = (4, 0.7, 0.286)
    param['retrain_freq'] = 500

    #simulation
    param['num_episodes'] = 10
    param['threshold'] = 0.1

    # transfer learning
    param['trans_timesteps_att'] = 1000
    param['trans_timesteps_def'] = 1000
    param['trans_lr'] = 5e-5
    param['trans_exploration_fraction'] = 0.5
    param['trans_exploration_final_eps'] = 0.03

    param['trans_exploration_fraction_att'] = 0.5
    param['trans_exploration_final_eps_att'] = 0.03
    param['trans_exploration_fraction_def'] = 0.5
    param['trans_exploration_final_eps_def'] = 0.03

    param_path = os.getcwd() + '/network_parameters/param.json'
    jp.save_json_data(param_path, param)
    print("Network parameters have been saved in a json file successfully.")


nn_param()