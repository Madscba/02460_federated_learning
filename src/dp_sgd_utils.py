import torch

def clip_gradients(net, optimizer, theta0, device):
    for layer, params in net.state_dict().items():
        diff = params - theta0[layer]
        params.data = theta0[layer] + per_layer_clip(diff, optimizer.max_grad_norm, device)

def add_noise(net, optimizer):
    for param in net.parameters():
        param.data += torch.normal(mean=0, std=optimizer.noise_multiplier)

def per_layer_clip(layer_gradient, max_grad_norm,device):
    total_norm = torch.norm(layer_gradient.data.detach(), p=1).to(device)
    clipped_gradient = layer_gradient * min(1, max_grad_norm / total_norm)
    return clipped_gradient