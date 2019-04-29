import os
from attackgraph import json_op as jp

def nn_param():
    param = {}
    param['num_hidden'] = 256
    param['num_layers'] = 1
    param['lr'] = 5e-5
    param['total_timesteps'] = 700000 #TODO: total time steps should be larger than hado time step.
    param['exploration_fraction'] = 0.5
    param['exploration_final_eps'] = 0.03
    param['print_freq'] = 250
    param['param_noise'] = False
    param['gamma'] = 0.99
    param['prioritized_replay'] = True
    param['checkpoint_freq'] = None

    #hado
    param['retrain_timesteps'] = 400000
    param['hado_param'] = (4, 0.7, 0.286)
    param['retrain_freq'] = 100000

    #simulation
    param['num_episodes'] = 100
    param['threshold'] = 0.1

    param_path = os.getcwd() + '/network_parameters/param.json'
    jp.save_json_data(param_path, param)
    print("Network parameters have been saved in a json file successfully.")

def nn_param1():
    param = {}
    param['num_hidden'] = 256
    param['num_layers'] = 1
    param['lr'] = 5e-5
    param['total_timesteps'] = 5000 #TODO: total time steps should be larger than hado time step.
    param['exploration_fraction'] = 0.5
    param['exploration_final_eps'] = 0.03
    param['print_freq'] = 250
    param['param_noise'] = False
    param['gamma'] = 0.99
    param['prioritized_replay'] = True
    param['checkpoint_freq'] = None

    #hado
    param['retrain_timesteps'] = 200
    param['hado_param'] = (4, 0.7, 0.286)
    param['retrain_freq'] = 1000

    #simulation
    param['num_episodes'] = 10
    param['threshold'] = 0.1

    param_path = os.getcwd() + '/network_parameters/param.json'
    jp.save_json_data(param_path, param)
    print("Network parameters have been saved in a json file successfully.")


nn_param1()