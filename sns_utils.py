import torch
import torch.nn as nn
from torch.nn import Parameter


class Non_Spiking_Step(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i_app, hidden, Gm, bm, Gmax, Esyn): 
        
        size = Gm.size()[0]
        
        # membrane leak, applied currents and bias currents
        step = -Gm*hidden + bm + i_app
        row_repeat_hidden = hidden.unsqueeze(2).expand(*hidden.size(),hidden.size(1))
        #rcol_repeat_hidden = hidden.unsqueeze(2).expand(*hidden.size(),hidden.size(1))
        # Calculating the synaptic currents
        diff_potential = Esyn - row_repeat_hidden
        diff_potential = torch.matmul(diff_potential,torch.clamp(row_repeat_hidden,min=0,max=1)*torch.eye(hidden.size(1)))
        diff_potential = diff_potential * Gmax
        diff_potential = torch.sum(diff_potential,dim=1)
        
        # adding membrane and synapse
        step = torch.add(step,diff_potential)
        
        # save the tensors 
        ctx.save_for_backward(i_app, hidden, Gm, bm, Gmax, Esyn, step)

        # euler step
        hidden = torch.add(hidden,step)
        
        return hidden

    @staticmethod
    def backward(ctx, grad_output):
        # load the saved tensors
        i_app, hidden, Gm, bm, Gmax, Esyn, step = ctx.saved_tensors
        row_repeat_hidden = hidden.unsqueeze(2).expand(*hidden.size(),hidden.size(1))
        clamped_hidden = torch.clamp(hidden,min=0,max=1)
        
        #calculating the trivial gradients
        grad_hidden = torch.ones(hidden.size())
        grad_input = grad_output
        grad_Gm = -grad_output*hidden
        grad_bm = grad_output
        
        # calculating the complicated weight gradients
        grad_Gmax = (torch.matmul(Esyn,torch.clamp(row_repeat_hidden,min=0,max=1)*torch.eye(hidden.size(1))) - torch.matmul(clamped_hidden.T,hidden)) * grad_output.unsqueeze(1)
        grad_Esyn = (torch.matmul(Gmax,torch.clamp(row_repeat_hidden,min=0,max=1)*torch.eye(hidden.size(1)))) * grad_output.unsqueeze(1)
        
        return grad_input, grad_hidden, grad_Gm, grad_bm, grad_Gmax, grad_Esyn


class SNSCell(nn.Module):
    def __init__(self, size):
        super(SNSCell, self).__init__()
        self.size = size
        self.Gm = Parameter(torch.rand(size))
        self.bm = Parameter(torch.rand(size))
        self.Gmax = Parameter(torch.rand(size,size))
        self.Esyn = Parameter(torch.rand(size,size))
        
        self.step = Non_Spiking_Step.apply
    
    def forward(self, i_app, hidden):
        with torch.no_grad():
            self.Gm[:] = self.Gm.clamp(0.01,1)
            self.bm[:] = self.bm.clamp(-1.0,1.0)
            self.Gmax[:] = self.Gmax.clamp(0,1)
            self.Esyn[:] = self.Esyn.clamp(-3,3)
        hidden = self.step(i_app, hidden, self.Gm, self.bm, self.Gmax, self.Esyn)
        #print(self.Gm)
        return hidden, hidden