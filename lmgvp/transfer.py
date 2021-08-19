# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: MIT-0

"""
Utils for transfer learning
"""
from torch.nn.parameter import Parameter


def load_state_dict_to_model(model, state_dict):
    """Initialize a model with parameters in `state_dict` (inplace)
    from a pretrained model with slightly different architecture.
    """
    own_state = model.state_dict()
    print("model own state keys:", len(own_state))
    print("state_dict keys:", len(state_dict))
    keys_loaded = 0
    for name, param in state_dict.items():
        if name not in own_state:
            continue
        if isinstance(param, Parameter):
            # backwards compatibility for serialized parameters
            param = param.data
        own_state[name].copy_(param)
        keys_loaded += 1
    print("keys loaded into model:", keys_loaded)
