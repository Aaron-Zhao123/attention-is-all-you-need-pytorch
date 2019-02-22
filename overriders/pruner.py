import numpy as np
import torch
import pickle
from torch.nn import Parameter


class Pruner(object):
    masks = {}

    def __init__(object, load_mask=None):
        if load_mask is not None:
            self._load_masks(load_mask)
        super(Pruner).__init__()

    def get_mask(self, value, name):
        mask = self.masks.get(name+'.mask')
        if mask is None:
            mask = Parameter(torch.ones(value.shape), requires_grad=False)
            self.masks[name+'.mask'] = value
            return mask
        else:
            return mask

    def update_masks(self, named_params, prune_params={'alpha':0.5}):
        for n, p in named_params:
            existing_mask = self.get_mask(p, n)
            mask = self._update_mask(p, existing_mask, params=prune_params)
            self.masks[n + '.mask'] = mask

    def _update_mask(self, value, existing_mask, params):
        # dns style update
        # alpha = params.get('alpha')
        # value = value.detach().numpy()
        # existing_mask = existing_mask.detach().numpy()
        # # threshold = np.percentile(value, alpha*100)
        # # mask = torch.Tensor((value > threshold).astype(np.float))
        # mean = np.mean(value)
        # std = np.std(value)
        # threshold = mean + alpha * std
        # on_mask = np.abs(value) > (1.1 * threshold)
        # off_mask = np.abs(value) > (0.9 * threshold)
        # new_mask = np.logical_or(existing_mask, on_mask)
        # new_mask = np.logical_and(new_mask, off_mask)
        # mask = torch.Tensor((new_mask).astype(np.float))
        # mask.requires_grad_(requires_grad=False)
        mask = (value > 0.02).float()
        return mask
        # return Parameter(mask, requires_grad=False)

    def _load_masks(self, fname):
        with open(fname, 'rb') as f:
            self.masks = pickle.load(f)

    def save_masks(self, fname):
        with open(fname, 'wb') as f:
            pickle.dump(fname, self.masks)




