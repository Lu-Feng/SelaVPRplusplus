import torch
from torch import nn
import torch.nn.functional as F
import math
from model.vision_transformer import vit_base, vit_large 
from model.aggregation import Flatten
from model.normalization import L2Norm
import model.aggregation as aggregation

class STE_binary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        p = (x >= 0) * (+1.0)
        n = (x < 0) * (-1.0)
        out = p + n
        return out
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * 1.0

class GeoLocalizationNet(nn.Module):
    """
    The used network is composed of a backbone and an aggregation layer to produce robust global representation.
    """
    def __init__(self, args):
        super().__init__()
        self.backbone = get_backbone(args)
        self.hashing = args.hashing
        self.rerank = args.rerank

        if args.backbone == "dinov2-base":
            input_dim = 768
            output_dim = 2048
        elif args.backbone == "dinov2-large":
            input_dim = 1024
            output_dim = 4096
        else:
            raise ValueError(f"Unknown backbone: {args.backbone}")

        # Training a vanilla single-branch VPR model or a two-branch model with reranking requires high-dimensional floating-point global features
        if not self.hashing or self.rerank:
            self.aggregation = get_aggregation(args)
            self.aggregation_name = args.aggregation
            if args.aggregation == "gem":
                self.aggregation = nn.Sequential(L2Norm(), self.aggregation, Flatten())
                self.linear1 = nn.Linear(input_dim, input_dim)
                self.linear2 = nn.Linear(input_dim, output_dim)
        
        # Training a hashing single-branch VPR model or a two-branch VPR model with reranking requires low-dimensional binary global features
        if self.hashing:
            self.aggregation_hashing = nn.Sequential(L2Norm(), aggregation.GeM(work_with_tokens=False), Flatten())
            self.linear3 = nn.Linear(input_dim, input_dim)
            self.linear4 = nn.Linear(input_dim, 512)
            
    def forward(self, x):
        x = self.backbone(x)    

        # Vanilla single-branch VPR model with floating-point global features
        if not self.hashing:
            B,P,D = x["x_prenorm"].shape
            W = H = int(math.sqrt(P-1))
            # GeM
            if self.aggregation_name == "gem":
                x_g = self.linear1(x["x_norm_patchtokens"].view(B,W,H,D)).permute(0,3,1,2)
                x_g = self.aggregation(x_g)
                x_g = self.linear2(x_g)
            # BoQ
            elif self.aggregation_name == "boq":
                x_g = self.aggregation(x["x_norm_patchtokens"].view(B,W,H,D).permute(0,3,1,2))
            # SALAD 
            elif self.aggregation_name == "salad":
                x_p = x["x_norm_patchtokens"].view(B,W,H,D).permute(0,3,1,2)
                x_c = x["x_norm_clstoken"]
                x_g = self.aggregation((x_p, x_c))
            # L2 normalization
            x_g = F.normalize(x_g, p=2, dim=-1)
            return x_g

        # Hashing single-branch VPR model with binary global features
        if self.hashing and not self.rerank:
            B,P,D = x["z_prenorm"].shape
            W = H = int(math.sqrt(P-1))
            z = self.linear3(x["z_norm_patchtokens"].view(B,W,H,D)).permute(0, 3, 1, 2) 
            z = self.aggregation_hashing(z)
            z = self.linear4(z)
            z = F.normalize(z, p=2, dim=-1)
            z1 = STE_binary.apply(z)
            return z, z1
        
        # Our novel Two-branch VPR model with reranking: low-dimensional binary global features for fast initial retrieval 
        # and high-dimensional floating-point global features for reranking
        if self.hashing and self.rerank:
            B,P,D = x["x_prenorm"].shape
            W = H = int(math.sqrt(P-1))
            # GeM
            if self.aggregation_name == "gem":
                x_g = self.linear1(x["x_norm_patchtokens"].view(B,W,H,D)).permute(0,3,1,2)
                x_g = self.aggregation(x_g)
                x_g = self.linear2(x_g)
            # BoQ
            elif self.aggregation_name == "boq":
                x_g = self.aggregation(x["x_norm_patchtokens"].view(B,W,H,D).permute(0,3,1,2))
            # SALAD 
            elif self.aggregation_name == "salad":
                x_p = x["x_norm_patchtokens"].view(B,W,H,D).permute(0,3,1,2)
                x_c = x["x_norm_clstoken"]
                x_g = self.aggregation((x_p, x_c))
            # L2 normalization
            x_g = F.normalize(x_g, p=2, dim=-1)

            z = self.linear3(x["z_norm_patchtokens"].view(B,W,H,D)).permute(0, 3, 1, 2) 
            z = self.aggregation_hashing(z)
            z = self.linear4(z)
            z = F.normalize(z, p=2, dim=-1)
            z1 = STE_binary.apply(z)

            return z, z1, x_g
        
def get_aggregation(args):
    if args.aggregation == "gem":
        args.work_with_tokens = False
        return aggregation.GeM(work_with_tokens=args.work_with_tokens)
    elif args.aggregation == "boq":
        if args.backbone == "dinov2-base":
            return aggregation.BoQ(in_channels=768, proj_channels=384, num_queries=64, num_layers=2, row_dim=32)
        elif args.backbone == "dinov2-large":
            return aggregation.BoQ(in_channels=1024, proj_channels=384, num_queries=64, num_layers=2, row_dim=32)
    elif args.aggregation == "salad":
        if args.backbone == "dinov2-base":
            return aggregation.SALAD(num_channels=768, num_clusters=64, cluster_dim=128, token_dim=256)
        elif args.backbone == "dinov2-large":
            return aggregation.SALAD(num_channels=1024, num_clusters=64, cluster_dim=128, token_dim=256)

def get_backbone(args):
    if args.backbone == "dinov2-base":
        backbone = vit_base(patch_size=14,img_size=518,init_values=1,block_chunks=0,hashing=args.hashing, rerank=args.rerank)
    elif args.backbone == "dinov2-large":
        backbone = vit_large(patch_size=14,img_size=518,init_values=1,block_chunks=0,hashing=args.hashing, rerank=args.rerank)

    if not args.resume:
        model_dict = backbone.state_dict()
        state_dict = torch.load(args.foundation_model_path)
        model_dict.update(state_dict.items())
        backbone.load_state_dict(model_dict)

    return backbone