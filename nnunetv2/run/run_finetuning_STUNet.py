from unittest.mock import patch
from run_training import run_training_entry
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import OrderedDict
import torch

def load_stunet_pretrained_weights(network, fname, verbose=False):

    saved_model = torch.load(fname)

    if fname.endswith('pt'):
        pretrained_dict = saved_model['network_weights']
    elif fname.endswith('model'):
        pretrained_dict = saved_model['state_dict']

    new_file3 = OrderedDict()
    for old_key, value in pretrained_dict.items():
        if 'encoder' in old_key:
            new_key = old_key.split('sp_cnn.')[-1]  # This extracts the part after the last '.'
            new_file3[new_key] = value

    if isinstance(network, DDP):
        mod = network.module
    else:
        mod = network
    if isinstance(mod, OptimizedModule):
        mod = mod._orig_mod

    mod_dict = mod.state_dict()
    # fun fact: in principle this allows loading from parameters that do not cover the entire network. For example pretrained
    # encoders. Not supported by this function though (see assertions above)

    # commenting out this abomination of a dict comprehension for preservation in the archives of 'what not to do'
    # pretrained_dict = {'module.' + k if is_ddp else k: v
    #                    for k, v in pretrained_dict.items()
    #                    if (('module.' + k if is_ddp else k) in model_dict) and
    #                    all([i not in k for i in skip_strings_in_pretrained])}

    for key, _ in mod_dict.items():
        if ('conv_blocks' in key):
            if (key in new_file3) and (mod_dict[key].shape == new_file3[key].shape):
                print('This layer worked: ', key)
            else:
                print('This layer not worked: ', key)

    mod.load_state_dict(new_file3, strict=False)

if __name__ == '__main__':
    with patch("run_training.load_pretrained_weights", load_stunet_pretrained_weights):
        run_training_entry()