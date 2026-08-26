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
    def forward(ctx, input, tau_f=1.0, tau_b=1.0):
        ctx.save_for_backward(input)
        ctx.tau_b = tau_b
        # Exploration forward pass (stochastic)
        p = torch.sigmoid(input / tau_f)
        output = torch.bernoulli(p) * 2.0 - 1.0
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        tau_b = ctx.tau_b
        # Gradient spread: sigmoid derivative scaled by backward temperature
        p_b = torch.sigmoid(input / tau_b)
        grad_input = grad_output * p_b * (1.0 - p_b) / tau_b
        return grad_input, None, None
