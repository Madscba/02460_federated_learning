from torch.nn import CrossEntropyLoss
from torch.nn import functional as F
from torch import Tensor, detach
import torch


class FedOptLoss(CrossEntropyLoss):

    def __init__(self,weight_0=None, weight=None, mu: float=0.1, 
                size_average=None, ignore_index: int = -100, reduce=None, reduction: str = 'mean') -> None:
        super().__init__(weight,size_average, ignore_index, reduce, reduction)
        self.weight_0, self.n_params=self.detach_weights_t(weight_0)
        self.mu=mu
        
    
    def forward(self, input: Tensor, target: Tensor, params = None) -> Tensor:
        return super().forward(input, target)+self.mu*self.proximal_term(params)

    def proximal_term(self, params):
        loss=0
        for  p, p_t in zip(params,self.weight_0):
            loss+=torch.sum((p-p_t)**2)
        sum_loss=loss
        return sum_loss
        
    def detach_weights_t(self, params):
        param_l=[]
        n_params=0
        for param in params:
            param_t=param.clone().detach().requires_grad_(False)
            param_l.append(param_t)
            n_params+=torch.flatten(param_t).shape[0]
        return param_l, n_params 
        
