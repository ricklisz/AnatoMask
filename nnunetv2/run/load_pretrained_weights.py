import torch
from torch._dynamo import OptimizedModule
from torch.nn.parallel import DistributedDataParallel as DDP
from collections import OrderedDict

def load_pretrained_weights(network, fname, verbose=False):
    """
    Transfers all weights between matching keys in state_dicts. matching is done by name and we only transfer if the
    shape is also the same. Segmentation layers (the 1x1(x1) layers that produce the segmentation maps)
    identified by keys ending with '.seg_layers') are not transferred!

    If the pretrained weights were optained with a training outside nnU-Net and DDP or torch.optimize was used,
    you need to change the keys of the pretrained state_dict. DDP adds a 'module.' prefix and torch.optim adds
    '_orig_mod'. You DO NOT need to worry about this if pretraining was done with nnU-Net as
    nnUNetTrainer.save_checkpoint takes care of that!

    """
    saved_model = torch.load(fname)
    pretrained_dict = saved_model['network_weights']

    skip_strings_in_pretrained = [
        '.seg_layers.',
    ]

    if isinstance(network, DDP):
        mod = network.module
    else:
        mod = network
    if isinstance(mod, OptimizedModule):
        mod = mod._orig_mod

    model_dict = mod.state_dict()
    # verify that all but the segmentation layers have the same shape
    for key, _ in model_dict.items():
        if all([i not in key for i in skip_strings_in_pretrained]):
            assert key in pretrained_dict, \
                f"Key {key} is missing in the pretrained model weights. The pretrained weights do not seem to be " \
                f"compatible with your network."
            assert model_dict[key].shape == pretrained_dict[key].shape, \
                f"The shape of the parameters of key {key} is not the same. Pretrained model: " \
                f"{pretrained_dict[key].shape}; your network: {model_dict[key]}. The pretrained model " \
                f"does not seem to be compatible with your network."

    # fun fact: in principle this allows loading from parameters that do not cover the entire network. For example pretrained
    # encoders. Not supported by this function though (see assertions above)

    # commenting out this abomination of a dict comprehension for preservation in the archives of 'what not to do'
    # pretrained_dict = {'module.' + k if is_ddp else k: v
    #                    for k, v in pretrained_dict.items()
    #                    if (('module.' + k if is_ddp else k) in model_dict) and
    #                    all([i not in k for i in skip_strings_in_pretrained])}

    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                       if k in model_dict.keys() and all([i not in k for i in skip_strings_in_pretrained])}

    model_dict.update(pretrained_dict)

    print("################### Loading pretrained weights from file ", fname, '###################')
    if verbose:
        print("Below is the list of overlapping blocks in pretrained model and nnUNet architecture:")
        for key, value in pretrained_dict.items():
            print(key, 'shape', value.shape)
        print("################### Done ###################")
    mod.load_state_dict(model_dict)

def load_stunet_ssl_weights(network, fname, verbose=False):

    saved_model = torch.load(fname)

    if fname.endswith('pt'):
        pretrained_dict = saved_model['network_weights']
    elif fname.endswith('model'):
        pretrained_dict = saved_model['state_dict']

    new_file3 = OrderedDict()
    for old_key, value in pretrained_dict.items():
        print(old_key)
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


def load_stunet_moco_weights(network, fname, verbose=False):

    saved_model = torch.load(fname)

    if fname.endswith('pt'):
        pretrained_dict = saved_model['network_weights']
    elif fname.endswith('model'):
        pretrained_dict = saved_model['state_dict']

    new_file3 = OrderedDict()
    for old_key, value in pretrained_dict.items():
        print(old_key)
        # if  'encoder' in old_key and 'conv_blocks_context' in old_key:
        #     new_key = old_key.split('encoder_')[1][4:]  # This extracts the part after the last '.'
        #     new_file3[new_key] = value
        if 'conv_blocks_context' in old_key:
            new_key = old_key.split('net')[1][3:]  # This extracts the part after the last '.'
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

def load_stunet_pretrained_weights(network, fname, verbose=False):

    saved_model = torch.load(fname, map_location=torch.device('cpu'))

    if fname.endswith('pth') or fname.endswith('pt'):
        pretrained_dict = saved_model['network_weights']
    elif fname.endswith('model'):
        pretrained_dict = saved_model['state_dict']

    skip_strings_in_pretrained = [
        'seg_outputs',
    ]

    if isinstance(network, DDP):
        mod = network.module
    else:
        mod = network
    if isinstance(mod, OptimizedModule):
        mod = mod._orig_mod

    model_dict = mod.state_dict()

    # Adjust weights for specific layers
    # verify that all but the segmentation layers have the same shape

    for key, _ in model_dict.items():
        if all([i not in key for i in skip_strings_in_pretrained]):
            assert key in pretrained_dict, \
                f"Key {key} is missing in the pretrained model weights. The pretrained weights do not seem to be " \
                f"compatible with your network."
            assert model_dict[key].shape == pretrained_dict[key].shape, \
                f"The shape of the parameters of key {key} is not the same. Pretrained model: " \
                f"{pretrained_dict[key].shape}; your network: {model_dict[key]}. The pretrained model " \
                f"does not seem to be compatible with your network."

    # fun fact: in principle this allows loading from parameters that do not cover the entire network. For example pretrained
    # encoders. Not supported by this function though (see assertions above)

    # commenting out this abomination of a dict comprehension for preservation in the archives of 'what not to do'
    # pretrained_dict = {'module.' + k if is_ddp else k: v
    #                    for k, v in pretrained_dict.items()
    #                    if (('module.' + k if is_ddp else k) in model_dict) and
    #                    all([i not in k for i in skip_strings_in_pretrained])}

    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                       if k in model_dict.keys() and all([i not in k for i in skip_strings_in_pretrained])}

    model_dict.update(pretrained_dict)

    print("################### Loading pretrained weights from file ", fname, '###################')
    if verbose:
        print("Below is the list of overlapping blocks in pretrained model and nnUNet architecture:")
        for key, value in pretrained_dict.items():
            print(key, 'shape', value.shape)
        print("################### Done ###################")
    mod.load_state_dict(model_dict)


def load_stunet_for_autopet(network, fname, verbose=False):
    saved_model = torch.load(fname, map_location=torch.device('cpu'))

    if fname.endswith('pth') or fname.endswith('pt'):
        pretrained_dict = saved_model['network_weights']
    elif fname.endswith('model'):
        pretrained_dict = saved_model['state_dict']

    skip_strings_in_pretrained = [
        'seg_outputs',
    ]

    if isinstance(network, DDP):
        mod = network.module
    else:
        mod = network
    if isinstance(mod, OptimizedModule):
        mod = mod._orig_mod

    model_dict = mod.state_dict()

    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                       if k in model_dict.keys() and all([i not in k for i in skip_strings_in_pretrained])}

    model_dict.update(pretrained_dict)

    model_dict['conv_blocks_context.0.0.conv1.weight'] = model_dict['conv_blocks_context.0.0.conv1.weight'].repeat(1, 2,1,1,1)
    model_dict['conv_blocks_context.0.0.conv3.weight'] = model_dict['conv_blocks_context.0.0.conv3.weight'].repeat(1, 2,1,1,1)

    print("################### Loading pretrained weights from file ", fname, '###################')
    if verbose:
        print("Below is the list of overlapping blocks in pretrained model and nnUNet architecture:")
        for key, value in pretrained_dict.items():
            print(key, 'shape', value.shape)
        print("################### Done ###################")
    mod.load_state_dict(model_dict)


def load_swin_weights(network, fname, verbose=False):
    """
    Transfers all weights between matching keys in state_dicts. matching is done by name and we only transfer if the
    shape is also the same. Segmentation layers (the 1x1(x1) layers that produce the segmentation maps)
    identified by keys ending with '.seg_layers') are not transferred!

    If the pretrained weights were optained with a training outside nnU-Net and DDP or torch.optimize was used,
    you need to change the keys of the pretrained state_dict. DDP adds a 'module.' prefix and torch.optim adds
    '_orig_mod'. You DO NOT need to worry about this if pretraining was done with nnU-Net as
    nnUNetTrainer.save_checkpoint takes care of that!

    """
    saved_model = torch.load(fname)
    pretrained_dict = saved_model['network_weights']

    skip_strings_in_pretrained = [
        'out.conv.conv.weight','out.conv.conv.bias'
    ]

    if isinstance(network, DDP):
        mod = network.module
    else:
        mod = network
    if isinstance(mod, OptimizedModule):
        mod = mod._orig_mod

    model_dict = mod.state_dict()
    # verify that all but the segmentation layers have the same shape
    for key, _ in model_dict.items():
        if all([i not in key for i in skip_strings_in_pretrained]):
            assert key in pretrained_dict, \
                f"Key {key} is missing in the pretrained model weights. The pretrained weights do not seem to be " \
                f"compatible with your network."
            assert model_dict[key].shape == pretrained_dict[key].shape, \
                f"The shape of the parameters of key {key} is not the same. Pretrained model: " \
                f"{pretrained_dict[key].shape}; your network: {model_dict[key]}. The pretrained model " \
                f"does not seem to be compatible with your network."

    # fun fact: in principle this allows loading from parameters that do not cover the entire network. For example pretrained
    # encoders. Not supported by this function though (see assertions above)

    # commenting out this abomination of a dict comprehension for preservation in the archives of 'what not to do'
    # pretrained_dict = {'module.' + k if is_ddp else k: v
    #                    for k, v in pretrained_dict.items()
    #                    if (('module.' + k if is_ddp else k) in model_dict) and
    #                    all([i not in k for i in skip_strings_in_pretrained])}

    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                       if k in model_dict.keys() and all([i not in k for i in skip_strings_in_pretrained])}

    model_dict.update(pretrained_dict)

    print("################### Loading pretrained weights from file ", fname, '###################')
    if verbose:
        print("Below is the list of overlapping blocks in pretrained model and nnUNet architecture:")
        for key, value in pretrained_dict.items():
            print(key, 'shape', value.shape)
        print("################### Done ###################")
    mod.load_state_dict(model_dict)

def load_unimiss_weights(network, fname, verbose=False):
    """
    Transfers all weights between matching keys in state_dicts. matching is done by name and we only transfer if the
    shape is also the same. Segmentation layers (the 1x1(x1) layers that produce the segmentation maps)
    identified by keys ending with '.seg_layers') are not transferred!

    If the pretrained weights were optained with a training outside nnU-Net and DDP or torch.optimize was used,
    you need to change the keys of the pretrained state_dict. DDP adds a 'module.' prefix and torch.optim adds
    '_orig_mod'. You DO NOT need to worry about this if pretraining was done with nnU-Net as
    nnUNetTrainer.save_checkpoint takes care of that!

    """
    saved_model = torch.load(fname)
    pretrained_dict = saved_model['network_weights']

    skip_strings_in_pretrained = ['cls_conv.weight', 'cls_conv.bias']

    # Determine if using DDP or DataParallel and adjust accordingly
    mod = network.module if isinstance(network, (DDP, torch.nn.DataParallel)) else network

    # Adjust for OptimizedModule if necessary
    if isinstance(mod, OptimizedModule):
        mod = mod._orig_mod

    model_dict = mod.state_dict()

    # Prepare the adjusted pretrained_dict
    pretrained_dict_adjusted = {}
    for k, v in pretrained_dict.items():
        adjusted_key = k  # Adjust key name if necessary, e.g., add or remove 'module.' prefix
        if all(skip_string not in adjusted_key for skip_string in skip_strings_in_pretrained):
            if adjusted_key in model_dict and model_dict[adjusted_key].shape == v.shape:
                pretrained_dict_adjusted[adjusted_key] = v

    # Update model's state_dict with the adjusted pretrained_dict
    mod.load_state_dict(pretrained_dict_adjusted, strict=False)

    if verbose:
        print("Loading pretrained weights from file:", fname)
        print("Overlapping blocks in pretrained model and current architecture:")
        for key in pretrained_dict_adjusted.keys():
            print(key, 'shape', pretrained_dict_adjusted[key].shape)
        print("Done loading pretrained weights.")
