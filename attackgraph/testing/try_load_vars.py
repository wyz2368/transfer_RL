# import tensorflow as tf
import joblib
import os
import copy

# sess = tf.Session()
# sess.__enter__()

# load_path = os.getcwd() + '/retrain_att/att_str_retrain2.pkl'
# dump_path = os.getcwd() + '/retrain_att/att_str_retrain2.pkl'
# loaded_params = joblib.load(os.path.expanduser(load_path))
# new_params = {}
# # print(type(loaded_params))
# keys = loaded_params.keys()
#
# for key in keys:
#     a = key.replace('att_str_retrain0','att_str_epoch4')
#     new_params[a] = loaded_params[key]
#
#
# joblib.dump(new_params, dump_path)



# load_path = os.getcwd() + '/attacker_strategies/att_str_epoch9.pkl'
load_path1 = os.getcwd() + '/defender_strategies/def_str_epoch2.pkl'
load_path2 = os.getcwd() + '/defender_strategies/def_str_epoch3.pkl'

loaded_params1 = joblib.load(os.path.expanduser(load_path1))
loaded_params2 = joblib.load(os.path.expanduser(load_path2))
for i in loaded_params1.keys():
    print(i)
    print(loaded_params1[i])
    print("______________________________")
    print(loaded_params1[i]-loaded_params2[i])
    print("*******************************************")