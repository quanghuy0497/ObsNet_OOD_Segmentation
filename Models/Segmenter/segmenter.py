import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F

from .utils import padding, unpadding
from timm.models.layers import trunc_normal_


class Segmenter(nn.Module):
    def __init__(
        self,
        encoder,
        decoder,
        n_cls,
        obsnet,
    ):
        super().__init__()
        self.n_cls = n_cls
        self.patch_size = encoder.patch_size
        self.encoder = encoder
        self.decoder = decoder
        self.obsnet = obsnet
        if self.obsnet:
            self.uncertainty = nn.Sequential(
                                nn.BatchNorm2d(n_cls),
                                nn.ReLU(),
                                nn.Conv2d(n_cls, 64, kernel_size=3, stride=1, padding=1, bias=False),
                                nn.BatchNorm2d(64),
                                nn.ReLU(),
                                nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1))

    @torch.jit.ignore
    def no_weight_decay(self):
        def append_prefix_no_weight_decay(prefix, module):
            return set(map(lambda x: prefix + x, module.no_weight_decay()))

        nwd_params = append_prefix_no_weight_decay("encoder.", self.encoder).union(
            append_prefix_no_weight_decay("decoder.", self.decoder))
        
        return nwd_params
    
    def forward(self, *args, **kwargs):
        if self.obsnet:
            return self.obsnet_forward(*args, **kwargs)
        else:
            return self.seg_forward(*args, **kwargs)

    def seg_forward(self, im, return_feat=False, **kwargs):
        H_ori, W_ori = im.size(2), im.size(3)
        im = padding(im, self.patch_size)
        H, W = im.size(2), im.size(3)

        x = self.encoder(im, return_feature=True)
        encoder = x

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        masks = self.decoder(x, (H, W))

        masks = F.interpolate(masks, size=(H, W), mode="bilinear")
        
        mask = unpadding(masks, (H_ori, W_ori))
        
        if return_feat:
            return [encoder, mask]
        return mask
    
    def obsnet_forward(self, im, return_feat=False, **kwargs):
        H_ori, W_ori = im.size(2), im.size(3)
        im = padding(im, self.patch_size)
        H, W = im.size(2), im.size(3)

        x = self.encoder(im, return_feature=True)
    
        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        masks = self.decoder(x, (H, W))

        masks = F.interpolate(masks, size=(H, W), mode="bilinear")
        
        masks = self.uncertainty(masks)
        
        return unpadding(masks, (H_ori, W_ori))

    def get_attention_map_enc(self, im, layer_id):
        return self.encoder.get_attention_map(im, layer_id)

    def get_attention_map_dec(self, im, layer_id):
        x = self.encoder(im, return_feat=True)

        # remove CLS/DIST tokens for decoding
        num_extra_tokens = 1 + self.encoder.distilled
        x = x[:, num_extra_tokens:]

        return self.decoder.get_attention_map(x, layer_id)
