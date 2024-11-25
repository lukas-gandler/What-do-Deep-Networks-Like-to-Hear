import warnings
import torch.nn as nn

from .combined_pipeline import CombinedPipeline
from .autoencoder.ArchiSound_audio_autoencoder import AudioAutoencoder

from .classifier.ESC50_MobileNet import get_mobilenet
from .classifier.ESC50_Dynamic_MobileNet import get_dynamic_mobilenet
from .classifier.ESC50_PaSST import get_passt


def get_classifier(classifier_name: str) -> nn.Module:
    if classifier_name == 'mn':
        print(f'=> Loading pretrained MobileNet model')
        return get_mobilenet(checkpoint='models/pretrained_weights/ESC50_mn10_esc50_epoch_79_acc_960.pt')
    elif classifier_name == 'dymn':
        print(f'=> Loading pretrained Dynamic-MobileNet model')
        return get_dynamic_mobilenet(checkpoint='models/pretrained_weights/ESC50_dymn10_esc50_epoch_79_acc_962.pt')
    elif classifier_name == 'passt':
        print(f'=> Loading pretrained PaSST model')

        warnings.filterwarnings('ignore')  # TODO look into warnings
        return get_passt(checkpoint='models/pretrained_weights/ESC50-passt-s-n-f128-p16-s10-fold1-acc.967.pt')
    else:
        raise RuntimeError(f'Invalid classifier name: {classifier_name}')