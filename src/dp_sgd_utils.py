import torch
import math

def clip_gradients(net, max_grad_norm, theta0, device):
    if max_grad_norm<100.0:
        for layer, params in net.state_dict().items():
            diff = params - theta0[layer]
            params.data = theta0[layer] + per_layer_clip(diff, max_grad_norm, device)

def add_noise(net, optimizer):
    sigma = (optimizer.noise_multiplier*optimizer.max_grad_norm)/optimizer.noise_scale
    for param in net.parameters():
        param.data += torch.normal(mean=0, std=torch.tensor(sigma))

def per_layer_clip(layer_gradient, max_grad_norm,device):
    clipping_value = max_grad_norm/math.sqrt(5)
    total_norm = torch.norm(layer_gradient.data.detach(), p=1).to(device)
    clipped_gradient = layer_gradient * min(1, clipping_value / total_norm)
    return clipped_gradient