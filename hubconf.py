dependencies = ['torch']

import torch
from model import network

class SimpleArgs:
    def __init__(self, **kwargs):
        self.backbone = kwargs.pop('backbone', 'dinov2-large')
        self.aggregation = kwargs.pop('aggregation', 'gem')

        self.hashing = kwargs.pop('hashing', True)
        self.rerank = kwargs.pop('rerank', True)

        self.resume = kwargs.pop("resume", True)
        self.foundation_model_path = kwargs.pop('foundation_model_path', None)
        
        for key, value in kwargs.items():
            setattr(self, key, value)

def SelaVPRplusplus(**kwargs):
    args = SimpleArgs(**kwargs)
    vpr_model = network.GeoLocalizationNet(args)
    vpr_model = torch.nn.DataParallel(vpr_model)
    if args.backbone == "dinov2-base":
      if not args.hashing:
        vpr_model.load_state_dict(
            torch.hub.load_state_dict_from_url(f'https://github.com/Lu-Feng/SelaVPRplusplus/releases/download/SelaVPR%2B%2B/SelaVPRplusplus_base.pth', map_location=torch.device('cpu'))["model_state_dict"]
        )
      elif args.hashing and args.rerank:
        vpr_model.load_state_dict(
            torch.hub.load_state_dict_from_url(f'https://github.com/Lu-Feng/SelaVPRplusplus/releases/download/SelaVPR%2B%2B/SelaVPRplusplus_base_rerank.pth', map_location=torch.device('cpu'))["model_state_dict"]
        )
    elif args.backbone == "dinov2-large":
      if not args.hashing:
        vpr_model.load_state_dict(
            torch.hub.load_state_dict_from_url(f'https://github.com/Lu-Feng/SelaVPRplusplus/releases/download/SelaVPR%2B%2B/SelaVPRplusplus_large.pth', map_location=torch.device('cpu'))["model_state_dict"]
        )
      elif args.hashing and args.rerank:
        vpr_model.load_state_dict(
            torch.hub.load_state_dict_from_url(f'https://github.com/Lu-Feng/SelaVPRplusplus/releases/download/SelaVPR%2B%2B/SelaVPRplusplus_large_rerank.pth', map_location=torch.device('cpu'))["model_state_dict"]
        )
    return vpr_model
