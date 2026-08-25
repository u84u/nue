import torch

class BinarySTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # Forward: use binary weights
        # sign(x) in {-1, 1}
        output = torch.sign(input)
        output[output == 0] = 1.0
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Backward: Identity (STE)
        return grad_output

def binary_quantize_ste(w):
    return BinarySTE.apply(w)

class DecoupledSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, tau_f):
        # Exploration forward pass (stochastic)
        p = torch.sigmoid(input / tau_f)
        output = torch.bernoulli(p) * 2.0 - 1.0
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Gradient spread backward pass
        return grad_output, None
