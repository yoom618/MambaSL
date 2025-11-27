import os
import torch
from models import DLinear, LightTS, MTSMixer, \
    TimesNet, ModernTCN, TimeMixerPP, \
    FEDformer, ETSformer, Crossformer, PatchTST, GPT4TS, iTransformer, \
    InterpretGN, MambaSimple, TSCMamba, \
    MambaSingleLayer, MambaMultiLayer

class Exp_Basic(object):
    def __init__(self, args):
        self.args = args
        self.model_dict = {
            ### referred to tslib
            'DLinear': DLinear,
            'LightTS': LightTS,
            'TimesNet': TimesNet,
            'FEDformer': FEDformer,
            'ETSformer': ETSformer,
            'Crossformer': Crossformer,
            'PatchTST': PatchTST,
            'iTransformer': iTransformer,
            'MambaSimple': MambaSimple,
            
            ### newly added (referred to the original repositories)
            'MTSMixer': MTSMixer,
            'ModernTCN': ModernTCN,
            'TimeMixerPP': TimeMixerPP,
            'GPT4TS': GPT4TS,
            'InterpretGN': InterpretGN,
            'TSCMamba': TSCMamba,
            'MambaSingleLayer': MambaSingleLayer,  # proposed
            'MambaMultiLayer': MambaMultiLayer,    # for ablation
        }
        if args.model == 'Mamba':
            print('Please make sure you have successfully installed mamba_ssm')
            from models import Mamba
            self.model_dict['Mamba'] = Mamba

        self.device = self.args.device
        self.model = self._build_model().to(self.device)

    def _build_model(self):
        raise NotImplementedError
        return None

    def _get_data(self):
        pass

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass
