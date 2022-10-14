import torch.nn as nn
import torch.nn.functional as F
import torch
from utils import ConjugateDualFunction


class fDALLoss(nn.Module):
    def __init__(self, divergence_name, gamma):
        super(fDALLoss, self).__init__()

        self.lhat = None
        self.phistar = None
        self.phistar_gf = None
        self.multiplier = 1.87  # Modified..... orginally 1.0
        self.internal_stats = {}
        self.domain_discriminator_accuracy = -1

        self.gammaw = gamma
        self.phistar_gf = lambda t: ConjugateDualFunction(divergence_name).fstarT(t)
        self.gf = lambda v: ConjugateDualFunction(divergence_name).T(v)

    def forward(self, y_s, y_t, y_s_adv, y_t_adv, K):
        # ---
        #
        #
        #print(y_s_adv.shape,'y_s avd shape')

        v_s = y_s_adv # h' output [4, 2, 200, 200]
        v_t = y_t_adv
        #temp_vs = v_s[:,0,:,:]

        # in fdal v_s is [128,10] before nll and after is [128] (batch = 128, class = 10)
        # same for v_t 
        # here y_s is [4,2,200,200]   h
        # 
        #print('before nll')
        #print(v_s)
        #print(v_s.shape,'bef sha',torch.max(v_s),'<<- max???')  [4, 2, 200, 200]  h'
        '''if K > 1:
            _, prediction_s = y_s.max(dim=1) # index which is max in h(x)   [4, 200, 200] 0 or 1
            _, prediction_t = y_t.max(dim=1)


            # This is not used here as a loss, it just a way to pick elements.

            # picking element prediction_s k element from y_s_adv.
            v_s = -F.nll_loss(v_s, prediction_s.detach(), reduction='none')  # vs = [4, 0, 200, 200] 0 if pred 0 otherwise 1
            # picking element prediction_t k element from y_t_adv.
            v_t = -F.nll_loss(v_t, prediction_t.detach(), reduction='none')
        '''
        #print('after pred_s')
        #print(prediction_s[0])
        #print(torch.bincount(prediction_s.flatten()),'bincount.....')
        #print('after nll')
        #print(v_s)
        #print(v_s.shape,'after sha')    [4, 200, 200]

        dst = torch.mean(torch.mean(self.gf(v_s), dim = (-2,-1))) - torch.mean(torch.mean(self.phistar_gf(v_t), dim = (-2,-1)))
        #print(v_s,'?????????vs max?>', torch.max(v_s))
        #print(torch.mean(torch.mean(self.gf(v_s), dim = (-2,-1))),'dst 1st term....')
        #print(torch.max(self.gf(v_s)),'gf max...')
        #print(torch.mean(torch.mean(self.phistar_gf(v_t), dim = (-2,-1))),'dst 2nd term....')
        #print(torch.max(self.phistar_gf(v_t)),'gfstar max...')

        self.internal_stats['lhatsrc'] = torch.mean(v_s).item()
        self.internal_stats['lhattrg'] = torch.mean(v_t).item()
        self.internal_stats['acc'] = self.domain_discriminator_accuracy
        self.internal_stats['dst'] = dst.item()

        # we need to negate since the obj is being minimized, so min -dst = max dst.
        # the gradient reversar layer will take care of the rest
        #print(dst,'dst term')
        return -self.multiplier * dst #multiplier = 1.87
