from attackgraph import json_op as jp
from baselines.common import models
from baselines.deepq.deepq import learn_multi_nets, Learner
import os

DIR_def = os.getcwd() + '/defender_strategies/'
DIR_att = os.getcwd() + '/attacker_strategies/'

def training_att(game, mix_str_def, epoch, retrain = False, transfer=False):
    if len(mix_str_def) != len(game.def_str):
        raise ValueError("The length of mix_str_def and def_str does not match while training")

    env = game.env
    env.reset_everything()

    env.set_training_flag(1)

    env.defender.set_mix_strategy(mix_str_def)
    env.defender.set_str_set(game.def_str)

    param_path = os.getcwd() + '/network_parameters/param.json'
    param = jp.load_json_data(param_path)

    if retrain:
        scope = 'att_str_retrain' + str(0) + '.pkl' + '/'
    else:
        scope = 'att_str_epoch' + str(1) + '.pkl' + '/'

    #TODO: Transfer Learning
    if epoch > 2:
        load_path = DIR_att + "att_str_epoch" + str(epoch-1) + ".pkl"
    else:
        load_path = None

    if transfer:
        lr = param['trans_lr']
        total_timesteps = param['trans_timesteps_att']
        ex_frac = param['trans_exploration_fraction']
        ex_final_eps = param['trans_exploration_final_eps']
    else:
        lr = param['lr']
        total_timesteps = param['total_timesteps_def']
        ex_frac = param['exploration_fraction']
        ex_final_eps = param['exploration_final_eps']

    learner = Learner(transfer=transfer)
    with learner.graph.as_default():
        with learner.sess.as_default():
            act_att, a_BD = learner.learn_multi_nets(
                env,
                network = models.mlp(num_hidden=param['num_hidden'], num_layers=param['num_layers']),
                lr =lr,
                total_timesteps=total_timesteps,
                exploration_fraction=ex_frac,
                exploration_final_eps=ex_final_eps,
                print_freq=param['print_freq'],
                param_noise=param['param_noise'],
                gamma=param['gamma'],
                prioritized_replay=param['prioritized_replay'],
                checkpoint_freq=param['checkpoint_freq'],
                scope = scope,
                load_path=load_path,
                epoch=epoch
            )
            print("Saving attacker's model to pickle.")
            if retrain:
                act_att.save(os.getcwd() + '/retrain_att/' + 'att_str_retrain' + str(0) + '.pkl', 'att_str_retrain' + str(0) + '.pkl' + '/')
            else:
                act_att.save(DIR_att + "att_str_epoch" + str(epoch) + ".pkl", 'att_str_epoch' + str(1) + '.pkl' + '/')
    learner.sess.close()
    return a_BD


def training_def(game, mix_str_att, epoch, retrain = False, transfer=False):
    if len(mix_str_att) != len(game.att_str):
        raise ValueError("The length of mix_str_att and att_str does not match while retraining")

    env = game.env
    env.reset_everything()

    env.set_training_flag(0)

    env.attacker.set_mix_strategy(mix_str_att)
    env.attacker.set_str_set(game.att_str)

    param_path = os.getcwd() + '/network_parameters/param.json'
    param = jp.load_json_data(param_path)

    if retrain:
        scope = 'def_str_retrain' + str(0) + '.pkl' + '/'
    else:
        scope = 'def_str_epoch' + str(0) + '.pkl' + '/'

    # TODO: Transfer Learning
    if epoch > 2:
        load_path = DIR_def + "def_str_epoch" + str(epoch-1) + ".pkl"
    else:
        load_path = None

    if transfer:
        lr = param['trans_lr']
        total_timesteps = param['trans_timesteps_def']
        ex_frac = param['trans_exploration_fraction']
        ex_final_eps = param['trans_exploration_final_eps']
    else:
        lr = param['lr']
        total_timesteps = param['total_timesteps_def']
        ex_frac = param['exploration_fraction']
        ex_final_eps = param['exploration_final_eps']

    learner = Learner(transfer=transfer)
    with learner.graph.as_default():
        with learner.sess.as_default():
            act_def, d_BD = learner.learn_multi_nets(
                env,
                network=models.mlp(num_hidden=param['num_hidden'], num_layers=param['num_layers']),
                lr=lr,
                total_timesteps=total_timesteps,
                exploration_fraction=ex_frac,
                exploration_final_eps=ex_final_eps,
                print_freq=param['print_freq'],
                param_noise=param['param_noise'],
                gamma=param['gamma'],
                prioritized_replay=param['prioritized_replay'],
                checkpoint_freq=param['checkpoint_freq'],
                scope = scope,
                load_path=load_path,
                epoch=-1
            )
            print("Saving defender's model to pickle.")
            if retrain:
                act_def.save(os.getcwd() + '/retrain_def/' + 'def_str_retrain' + str(0) + '.pkl', 'def_str_retrain' + str(0) + '.pkl' + '/')
            else:
                act_def.save(DIR_def + "def_str_epoch" + str(epoch) + ".pkl", "def_str_epoch" + str(0) + '.pkl' + '/')
    learner.sess.close()
    return d_BD

# for all strategies learned by retraining, the scope index is 0.
# TODO: Have not been modified for transfer learning.
def training_hado_att(game, transfer=False):
    param = game.param
    mix_str_def = game.hado_str(identity=0, param=param)

    if len(mix_str_def) != len(game.def_str):
        raise ValueError("The length of mix_str_def and def_str does not match while retraining")

    env = game.env
    env.reset_everything()

    env.set_training_flag(1)

    env.defender.set_mix_strategy(mix_str_def)
    env.defender.set_str_set(game.def_str)

    param_path = os.getcwd() + '/network_parameters/param.json'
    param = jp.load_json_data(param_path)

    if transfer:
        lr = param['trans_lr']
        total_timesteps = param['trans_timesteps']
        ex_frac = param['trans_exploration_fraction']
        ex_final_eps = param['trans_exploration_final_eps']
    else:
        lr = param['lr']
        total_timesteps = param['total_timesteps']
        ex_frac = param['exploration_fraction']
        ex_final_eps = param['exploration_final_eps']

    learner = Learner(retrain=True, freq=param['retrain_freq'])
    #TODO: add epoch???
    with learner.graph.as_default():
        with learner.sess.as_default():
            act_att,_ = learner.learn_multi_nets(
                env,
                network = models.mlp(num_hidden=param['num_hidden'], num_layers=param['num_layers']),
                lr = lr,
                total_timesteps=total_timesteps,
                exploration_fraction=ex_frac,
                exploration_final_eps=ex_final_eps,
                print_freq=param['print_freq'],
                param_noise=param['param_noise'],
                gamma=param['gamma'],
                prioritized_replay=param['prioritized_replay'],
                checkpoint_freq=param['checkpoint_freq'],
                scope = 'att_str_retrain' + str(0) + '.pkl' + '/',
                load_path=os.getcwd() + '/retrain_att/' + 'att_str_retrain' + str(0) + '.pkl'
            )
            # print("Saving attacker's model to pickle.")
            # act_att.save(os.getcwd() + '/retrain_att/' + 'att_str_retrain' + str(epoch) + ".pkl", 'att_str_epoch' + str(epoch) + '.pkl' + '/')
    learner.sess.close()


def training_hado_def(game, transfer=False):
    param = game.param
    mix_str_att = game.hado_str(identity=1, param=param)

    if len(mix_str_att) != len(game.att_str):
        raise ValueError("The length of mix_str_att and att_str does not match while training")

    env = game.env
    env.reset_everything()

    env.set_training_flag(0)

    env.attacker.set_mix_strategy(mix_str_att)
    env.attacker.set_str_set(game.att_str)

    param_path = os.getcwd() + '/network_parameters/param.json'
    param = jp.load_json_data(param_path)

    if transfer:
        lr = param['trans_lr']
        total_timesteps = param['trans_timesteps']
        ex_frac = param['trans_exploration_fraction']
        ex_final_eps = param['trans_exploration_final_eps']
    else:
        lr = param['lr']
        total_timesteps = param['total_timesteps']
        ex_frac = param['exploration_fraction']
        ex_final_eps = param['exploration_final_eps']

    learner = Learner(retrain=True, freq=param['retrain_freq'])
    with learner.graph.as_default():
        with learner.sess.as_default():
            act_def,_ = learner.learn_multi_nets(
                env,
                network=models.mlp(num_hidden=param['num_hidden'], num_layers=param['num_layers']),
                lr=lr,
                total_timesteps=total_timesteps,
                exploration_fraction=ex_frac,
                exploration_final_eps=ex_final_eps,
                print_freq=param['print_freq'],
                param_noise=param['param_noise'],
                gamma=param['gamma'],
                prioritized_replay=param['prioritized_replay'],
                checkpoint_freq=param['checkpoint_freq'],
                scope = 'def_str_retrain' + str(0) + '.pkl' + '/',
                load_path = os.getcwd() + '/retrain_def/' + 'def_str_retrain' + str(0) + '.pkl'
            )
            # print("Saving defender's model to pickle.")
            # act_def.save(os.getcwd() + '/retrain_def/' + 'def_str_retrain' + str(epoch) + ".pkl", "def_str_epoch" + str(epoch) + '.pkl' + '/')
    learner.sess.close()