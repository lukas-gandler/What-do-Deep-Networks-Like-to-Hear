import warnings

import torch
import torch.nn as nn

from .combined_pipeline import CombinedPipeline
from .autoencoder.ArchiSound_audio_autoencoder import AudioAutoencoder

from .classifier.ESC50_MobileNet import get_mobilenet
from .classifier.ESC50_Dynamic_MobileNet import get_dynamic_mobilenet
from .classifier.ESC50_PaSST import get_passt



mn_pretrained_weights = {
    'mn': {'checkpoint': 'models/pretrained_weights/ESC50_mn10_esc50_epoch_79_acc_960.pt'},
    'mn_rr1': {'checkpoint': 'models/pretrained_weights/ESC50-mn10_esc50_epoch_79_acc_950_rr1.pt'},
    'mn_rr2': {'checkpoint': 'models/pretrained_weights/ESC50-mn10_esc50_epoch_79_acc_960_rr2.pt'},
}



dymn_pretrained_weights = {
    'dymn':        {'checkpoint': 'models/pretrained_weights/ESC50_dymn10_esc50_epoch_79_acc_962.pt'},
    'dymn_rr1':    {'checkpoint': 'models/pretrained_weights/ESC50-dymn10_esc50_epoch_79_acc_960_rr1.pt'},
    'dymn_rr2':    {'checkpoint': 'models/pretrained_weights/ESC50-dymn10_esc50_epoch_79_acc_958_rr2.pt'},

    'dymn_noCA':   {'checkpoint': 'models/pretrained_weights/ESC50-dymn10_esc50_epoch_79_acc_870_no_CA.pt', 'no_ca': True},
    'dymn_noDC':   {'checkpoint': 'models/pretrained_weights/ESC50-dymn10_esc50_epoch_79_acc_298_no_DC.pt', 'no_dyconv': True},
    'dymn_noDR':   {'checkpoint': 'models/pretrained_weights/ESC50-dymn10_esc50_epoch_79_acc_892_no_DR.pt', 'no_dyrelu': True},

    'dymn_onlyCA': {'checkpoint': 'models/pretrained_weights/ESC50-dymn10_esc50_epoch_79_acc_375_only_CA.pt', 'no_dyconv': True, 'no_dyrelu': True},
    'dymn_onlyDC': {'checkpoint': 'models/pretrained_weights/ESC50-dymn10_esc50_epoch_79_acc_785_only_DC.pt', 'no_dyrelu': True, 'no_ca': True},
    'dymn_onlyDR': {'checkpoint': 'models/pretrained_weights/ESC50-dymn10_esc50_epoch_79_acc_340_only_DR.pt', 'no_dyconv': True, 'no_ca': True},
}

passt_pretrained_weights = {
    'passt': {'checkpoint': 'models/pretrained_weights/ESC50-passt-s-n-f128-p16-s10-fold1-acc.967.pt'},
}

def reset_weights(layer: nn.Module) -> None:
    if isinstance(layer, (nn.Conv1d, nn.ConvTranspose1d, nn.Linear)):
        nn.init.xavier_uniform_(layer.weight)
        if layer.bias is not None:
            nn.init.zeros_(layer.bias)
    elif isinstance(layer, nn.GroupNorm):
        nn.init.ones_(layer.weight)
        nn.init.zeros_(layer.bias)

def get_classifier(classifier_name: str) -> nn.Module:
    if classifier_name in mn_pretrained_weights:
        print(f'=> Loading pretrained MobileNet - {classifier_name}')
        mn_configuration = mn_pretrained_weights[classifier_name]
        return get_mobilenet(**mn_configuration)
    elif classifier_name in dymn_pretrained_weights:
        print(f'=> Loading pretrained Dynamic MobileNet - {classifier_name}')
        dymn_configuration = dymn_pretrained_weights[classifier_name]
        return get_dynamic_mobilenet(**dymn_configuration)
    elif classifier_name in passt_pretrained_weights:
        print(f'=> Loading pretrained PaSSt - {classifier_name}')
        warnings.filterwarnings('ignore')  # passt will throw warnings because it was trained on longer waveforms than 5 seconds but this won't affect performance
        pass_configuration = passt_pretrained_weights[classifier_name]
        return get_passt(**pass_configuration)
    else:
        raise RuntimeError(f'Invalid classifier name: {classifier_name}')


    #if classifier_name == 'mn':
    #    print(f'=> Loading pretrained MobileNet model')
    #    return get_mobilenet(checkpoint='models/pretrained_weights/ESC50_mn10_esc50_epoch_79_acc_960.pt')
    #elif classifier_name == 'mn_rr1':
    #    print(f'=> Loading pretrained MobileNet model from rr1')
    #    return get_mobilenet(checkpoint='models/pretrained_weights/ESC50-mn10_esc50_epoch_79_acc_950_rr1.pt')
    #elif classifier_name == 'dymn':
    #    print(f'=> Loading pretrained Dynamic-MobileNet model')
    #    return get_dynamic_mobilenet(checkpoint='models/pretrained_weights/ESC50_dymn10_esc50_epoch_79_acc_962.pt')
    #elif classifier_name == 'dymn_rr1':
    #    print(f'=> Loading pretrained Dynamic-MobileNet model from rr1')
    #    return get_dynamic_mobilenet(checkpoint='models/pretrained_weights/ESC50-dymn10_esc50_epoch_79_acc_960_rr1.pt')
    #elif classifier_name == 'dymn_noCA':
    #    print(f'=> Loading pretrained Dynamic-MobileNet model without Coordinate Attention')
    #    return get_dynamic_mobilenet(checkpoint='models/pretrained_weights/ESC50-dymn10_esc50_epoch_79_acc_870_no_CA.pt', no_ca=True)
    #elif classifier_name == 'dymn_noDC':
    #    print(f'=> Loading pretrained Dynamic-MobileNet model without Dynamic Convolution')
    #    return get_dynamic_mobilenet(checkpoint='models/pretrained_weights/ESC50-dymn10_esc50_epoch_79_acc_298_no_DC.pt', no_dyconv=True)
    #elif classifier_name == 'dymn_noDR':
    #    print(f'=> Loading pretrained Dynamic-MobileNet model without Dynamic ReLU')
    #    return get_dynamic_mobilenet(checkpoint='models/pretrained_weights/ESC50-dymn10_esc50_epoch_79_acc_892_no_DR.pt', no_dyrelu=True)
    #elif classifier_name == 'dymn_onlyCA':
    #    print(f'=> Loading pretrained Dynamic-MobileNet model with only Coordinate Attention')
    #    return get_dynamic_mobilenet(checkpoint='models/pretrained_weights/ESC50-dymn10_esc50_epoch_79_acc_375_only_CA.pt', no_dyconv=True, no_dyrelu=True)
    #elif classifier_name == 'dymn_onlyDC':
    #    print(f'=> Loading pretrained Dynamic-MobileNet model with only Dynamic Convolution')
    #    return get_dynamic_mobilenet(checkpoint='models/pretrained_weights/ESC50-dymn10_esc50_epoch_79_acc_785_only_DC.pt', no_dyrelu=True, no_ca=True)
    #elif classifier_name == 'dymn_onlyDR':
    #    print(f'=> Loading pretrained Dynamic-MobileNet model with only Dynamic ReLU')
    #    return get_dynamic_mobilenet(checkpoint='models/pretrained_weights/ESC50-dymn10_esc50_epoch_79_acc_340_only_DR.pt', no_dyconv=True, no_ca=True)
    #elif classifier_name == 'passt':
    #    print(f'=> Loading pretrained PaSST model')
    #
    #    warnings.filterwarnings('ignore')  # passt will throw warnings because it was trained on longer waveforms than 5 seconds but this won't affect performance
    #    return get_passt(checkpoint='models/pretrained_weights/ESC50-passt-s-n-f128-p16-s10-fold1-acc.967.pt')
    #else:
    #    raise RuntimeError(f'Invalid classifier name: {classifier_name}')

def get_autoencoder(autoencoder_type: str, keep_channel_dim: bool, mono_output: bool=True) -> nn.Module:
    if autoencoder_type == 'esc-pretrained':
        print(f'=> Loading pretrained ESC-50 autoencoder')

        autoencoder = AudioAutoencoder(keep_channel_dim=keep_channel_dim, mono_output=mono_output)
        autoencoder.load_state_dict(torch.load(f'../experiment-results/AE-pretraining/AE-ESC50-12 Experiment/final_model.pth', map_location='cpu')['model_state_dict'])
        return autoencoder
    elif autoencoder_type == 'archisound':
        print(f'=> Loading Archisound autoencoder')

        autoencoder = AudioAutoencoder(keep_channel_dim=keep_channel_dim, mono_output=mono_output)
        return autoencoder
    elif autoencoder_type == 'random':
        print(f'=> Loading randomly initialized autoencoder')

        autoencoder = AudioAutoencoder(keep_channel_dim=keep_channel_dim, mono_output=mono_output)
        autoencoder.apply(reset_weights)
        return autoencoder
    else:
        raise RuntimeError(f'Invalid autoencoder type: {autoencoder_type}')
