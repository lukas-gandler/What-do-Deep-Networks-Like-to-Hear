import sys
import os
import torch
import torch.nn as nn

from hear21passt.base import get_basic_model, get_model_passt

def get_passt(checkpoint: str=None) -> nn.Module:

    # Note: as the hear21passt library is printing the entire model architecture to the console when you load it (which is quite annoying)
    # we redirect the std-output here just for this bit such that it does not show up
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, 'w')

    model = get_basic_model(mode='logits')
    model.net = get_model_passt(arch='passt_s_swa_p16_128_ap476', n_classes=50)

    if checkpoint is not None:
        state_dict = torch.load(checkpoint, map_location='cpu', weights_only=True)
        model.net.load_state_dict(state_dict)

    sys.stdout = old_stdout
    return model