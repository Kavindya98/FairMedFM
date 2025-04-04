import torch
import torch.nn as nn
import torch.nn.functional as F

from wrappers.base import BaseWrapper


class UnSupCLIPWrapper(BaseWrapper):
    def __init__(self, model, cls_text_features, att_text_features, *args, **kwargs) -> None:
        super().__init__(model, *args, **kwargs)

        self.model = model
        self.cls_text_features = cls_text_features
        self.att_text_features = att_text_features
       
        self.cls_prototypes = nn.Parameter(cls_text_features.clone())
        self.att_prototypes = nn.Parameter(att_text_features.clone())

        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.model.forward_clip(x, self.cls_prototypes)
    
    def forward_att(self, x):
        return self.model.forward_clip(x, self.att_prototypes)

    def load_cls_prototype(self, cls_prototype):
        device = self.cls_prototypes.device
        self.cls_prototypes = nn.Parameter(cls_prototype.clone().to(device))
    
    def load_att_prototype(self, att_prototype):
        device = self.att_prototypes.device
        self.att_prototypes = nn.Parameter(att_prototype.clone().to(device))

    def get_cls_prototype(self):
        return self.cls_prototypes
    
    def get_att_prototype(self):
        return self.att_prototypes
