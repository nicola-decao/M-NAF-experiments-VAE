import torch
import math
import numpy as np


def logsumexp(inputs, dim=-1, keepdim=False):
    try: 
        return torch.logsumexp(inputs, dim=dim, keepdim=keepdim)
    except:
        m = inputs.max(dim=dim, keepdim=True)[0]
        return m.squeeze(dim) + (inputs - m).exp().sum(dim=dim, keepdim=keepdim).log()
    
class Sequential(torch.nn.Sequential):

    def logdetj(self, inputs=None):
        logdetj = 0.
        for module in self._modules.values():
            logdetj = logdetj + module.logdetj()
        return logdetj
    
    
class NNFlow(torch.nn.Sequential):
    
    def logdetj(self, inputs=None):
        for module in self._modules.values():
            inputs = module.log_diag_jacobian(inputs)
            inputs = inputs if len(inputs.shape) == 4 else inputs.view(inputs.shape + [1, 1])
        return inputs.squeeze().sum(-1)
    
    
class ResNNFlow(torch.nn.Sequential):
    
    def __init__(self, *args, **kwargs):
        super(ResNNFlow, self).__init__(*args, **kwargs)
        self.gate = torch.nn.Parameter(torch.nn.init.normal_(torch.Tensor(1)))
    
    def forward(self, inputs):
        or_inputs = inputs
        for module in self._modules.values():
            inputs = module(inputs)
#         return inputs + or_inputs
        return self.gate.sigmoid() * inputs + (1 - self.gate.sigmoid()) * or_inputs
    
    def logdetj(self, inputs=None):
        for module in self._modules.values():
            inputs = module.log_diag_jacobian(inputs)
            inputs = inputs if len(inputs.shape) == 4 else inputs.view(inputs.shape + [1, 1])
            
#         return torch.nn.functional.softplus(inputs.squeeze()).sum(-1)
        return (torch.nn.functional.softplus(grad.squeeze() + self.gate) - \
             torch.nn.functional.softplus(self.gate)).sum(-1)
    
class ConditionalResNNFlow(NNFlow):
    
    def logdetj(self, inputs=None):
        for module in self._modules.values():
            inputs = module.log_diag_jacobian(inputs)
            inputs = inputs if len(inputs.shape) == 4 else inputs.view(inputs.shape + [1, 1])
            
#         return logsumexp(torch.stack((inputs.squeeze(), torch.zeros_like(inputs.squeeze())), -1), -1).sum(-1)
        return torch.nn.functional.softplus(inputs.squeeze()).sum(-1)
    
    def forward(self, conditions, inputs):
        or_inputs = inputs
        i = 0
        for module in self._modules.values():
            if isinstance(module, MaskedConditionalWeight):
                inputs = module(conditions[i], inputs)
                i += 1
            else:
                inputs = module(inputs)
        return inputs + or_inputs
    
class Permutation(torch.nn.Module):
    
    def __init__(self, in_features, p=None):
        super(Permutation, self).__init__()
        self.in_features = in_features
        self.p = np.random.permutation(in_features) if p is None else p
        
    def forward(self, inputs):
        return inputs[:,self.p]
    
    def logdetj(self):
        return 0
    
    def __repr__(self):
        return 'Permutation(in_features={}, p={})'.format(
            self.in_features, self.p)
    

class MaskedWeight(torch.nn.Module):

    def __init__(self, in_features, out_features, dim, fill_upper=0, bias=True):
        super(MaskedWeight, self).__init__()
        self.in_features, self.out_features, self.dim, self.fill_upper, self.bias = in_features,\
            out_features, dim, fill_upper, bias
        self.init = torch.nn.init.normal

        weights = [self.init(torch.Tensor(out_features // dim, (i + 1) * in_features // dim))
                   for i in range(self.dim)]

        weight = torch.zeros(out_features, in_features)
        for i in range(self.dim):
            weight[i * out_features // dim:(i + 1) * out_features // dim,
                   0:(i + 1) * in_features // dim] = weights[i]
            
        self.weight = torch.nn.Parameter(weight * 0.1)
        self.bias = torch.nn.Parameter(torch.zeros(out_features)) if bias else 0
        
        self.mask_diag = torch.zeros_like(self.weight)
        for i in range(self.dim):
            self.mask_diag[i * (self.out_features // self.dim):(i + 1) * (self.out_features // self.dim),
                           i * (self.in_features // self.dim):(i + 1) * (self.in_features // self.dim)] = 1
            
        self.mask_upper = torch.ones_like(self.weight)
        for i in range(self.dim - 1):
            self.mask_upper[i * (self.out_features // self.dim):(i + 1) * (self.out_features // self.dim),
                            (i + 1) * (self.in_features // self.dim):] = self.fill_upper
            
        self.diag_w = torch.nn.Parameter(torch.rand(self.out_features, 1).log())
    
    def create_w(self):
        w = self.mask_diag * torch.exp(self.weight) + (1 - self.mask_diag) * self.weight * self.mask_upper
        
        self.norm_w = (w ** 2).sum(-1, keepdim=True)
        
        w = self.diag_w.exp() * w / self.norm_w.sqrt()
        
        return w.t()
    
    def create_wpl(self):
        wpl = self.diag_w + self.weight - 0.5 * torch.log(self.norm_w) 
        
        return wpl.t()[self.mask_diag.byte().t()].view(
            self.dim, self.in_features // self.dim, self.out_features // self.dim)
    
    def forward(self, inputs):

        if torch.cuda.is_available():
            self.mask_diag = self.mask_diag.cuda()
            self.mask_upper = self.mask_upper.cuda()

        self.tmp_w = self.create_w()
        self.tmp_wpl = self.create_wpl()
        
        self.tmp_inputs = inputs
        return self.tmp_inputs.matmul(self.tmp_w) + self.bias

    def jacobian(self, grad=None):
        g = self.tmp_w.t().unsqueeze(0).repeat(self.tmp_inputs.shape[0], 1, 1)
        return (g.matmul(grad)) if grad is not None else g
    
    def log_diag_jacobian(self, grad=None):
        g = self.tmp_wpl.transpose(-2, -1).unsqueeze(0).repeat(self.tmp_inputs.shape[0], 1, 1, 1)
        return logsumexp(g.unsqueeze(-2) + grad.transpose(-2, -1).unsqueeze(-3), -1) if grad is not None else g
    
    def __repr__(self):
        return 'MaskedWeight(in_features={}, out_features={}, dim={}, fill_upper={}, bias={})'.format(
            self.in_features, self.out_features, self.dim, self.fill_upper, not isinstance(self.bias, int))

    
class ConditionalNNFlow(NNFlow):
    
    def forward(self, conditions, inputs):
        i = 0
        for module in self._modules.values():
            if isinstance(module, MaskedConditionalWeight):
                inputs = module(conditions[i], inputs)
                i += 1
            else:
                inputs = module(inputs)
        return inputs
    
    
class ConditionalSequential(torch.nn.Sequential):
    
    def logdetj(self, inputs=None):
        logdetj = 0.
        for module in self._modules.values():
            logdetj = logdetj + module.logdetj()
        return logdetj
    
    def forward(self, conditions, inputs):
        i = 0
        for module in self._modules.values():
            if isinstance(module, ConditionalNNFlow) or isinstance(module, ConditionalResNNFlow):
                inputs = module(conditions[i], inputs)
                i += 1
            else:
                inputs = module(inputs)
        return inputs
    
    
class MaskedConditionalWeight(MaskedWeight):
    
    def __init__(self, in_features, out_features, dim, fill_upper=0, bias=True):
        super(MaskedConditionalWeight, self).__init__(in_features, out_features, dim, fill_upper, bias)
        
    def create_w(self, cond_inputs1, cond_inputs2):
        w = cond_inputs1 + self.weight.unsqueeze(0)
        
        w = self.mask_diag * torch.exp(w) + (1 - self.mask_diag) * (w) * self.mask_upper
        
        self.norm_w = (w ** 2).sum(-1, keepdim=True)
        
        w = torch.exp(cond_inputs2 + self.diag_w.unsqueeze(0)) * w / self.norm_w.sqrt()
        
        return w.transpose(-1, -2)
    
    def create_wpl(self, cond_inputs1, cond_inputs2):
        wpl = cond_inputs2 + self.diag_w + cond_inputs1 + self.weight - 0.5 * torch.log(self.norm_w) 
        
        return torch.masked_select(wpl.transpose(-1, -2), self.mask_diag.byte().transpose(-1, -2)).view(
            -1, self.dim, self.in_features // self.dim, self.out_features // self.dim)
            
    def forward(self, condition, inputs):
        cond_inputs1 = condition[:, :self.in_features].unsqueeze(1)
        cond_inputs2 = condition[:, self.in_features:self.in_features+self.out_features].unsqueeze(2)
        cond_inputs3 = condition[:, self.in_features+self.out_features:self.in_features+self.out_features+self.out_features]
        
        if torch.cuda.is_available():
            self.mask_diag = self.mask_diag.cuda()
            self.mask_upper = self.mask_upper.cuda()
                        
        self.tmp_w = self.create_w(cond_inputs1, cond_inputs2)
        self.tmp_wpl = self.create_wpl(cond_inputs1, cond_inputs2)
        
        return torch.matmul(inputs.unsqueeze(1), self.tmp_w).squeeze(1) + self.bias + cond_inputs3
    
    def log_diag_jacobian(self, grad=None):
        g = self.tmp_wpl.transpose(-2, -1)
        return logsumexp(g.unsqueeze(-2) + grad.transpose(-2, -1).unsqueeze(-3), -1) if grad is not None else g
    
    def __repr__(self):
        return 'MaskedConditionalWeight(in_features={}, out_features={}, dim={}, fill_upper={}, bias={})'.format(
            self.in_features, self.out_features, self.dim, self.fill_upper, not isinstance(self.bias, int))


class Tanh(torch.nn.Module):

    def forward(self, inputs):
        self.tmp_inputs = inputs
        return torch.tanh(self.tmp_inputs)

    def log_diag_jacobian(self, grad=None):
        g = - 2 * (self.tmp_inputs - math.log(2) + torch.nn.functional.softplus(- 2 * self.tmp_inputs))
        return (g.view(grad.shape) + grad) if grad is not None else g
    
    def logdetj(self):
        g = - 2 * (self.tmp_inputs - math.log(2) + torch.nn.functional.softplus(- 2 * self.tmp_inputs))
        return g.sum(-1)
    
# class HouseholderLinear(torch.nn.Module):
    
#     def __init__(self, in_features, k=None, bias=True):
#         super().__init__()
#         k = k if k is not None else in_features
#         self.in_features, self.k = in_features, k
#         self.weight_ = torch.nn.Parameter(torch.randn(k, in_features, 1))
#         self.weight_diag = torch.nn.Parameter(torch.rand(in_features, 1).log())
#         self.bias = torch.nn.Parameter(torch.zeros(in_features)) if bias else 0

#     @property
#     def weight_orth(self):
#         t = 2 * (self.weight_ * self.weight_.transpose(-1, -2)) / (self.weight_ ** 2).sum((-1, -2), keepdim=True)
#         return torch.chain_matmul(*torch.eye(self.in_features).to(t.device) - t)

#     @property
#     def weight(self):
#         return self.weight_diag.exp() * self.weight_orth

#     def forward(self, inputs):
#         return inputs @ self.weight + self.bias

#     def logdetj(self):
#         return self.weight_diag.sum()
    
    
# class ProjectedLinear(torch.nn.Module):
#     def __init__(self, in_features, bias=True, diag=True):
#         super().__init__()
#         self.in_features = in_features
#         self.weight_ = torch.nn.Parameter(torch.randn(in_features, in_features))
#         self.weight_diag = torch.nn.Parameter(torch.rand(in_features, 1).log()) if diag else 1
#         self.bias = torch.nn.Parameter(torch.zeros(in_features)) if bias else 0

#     @property
#     def weight_orth(self):
#         u, _, v = torch.svd(self.weight_)
#         return u @ v

#     @property
#     def weight(self):
#         return self.weight_diag.exp() * self.weight_orth

#     def forward(self, inputs):
#         return inputs @ self.weight + self.bias

#     def logdetj(self):
#         return self.weight_diag.sum() if not isinstance(self.weight_diag, int) else 0
    