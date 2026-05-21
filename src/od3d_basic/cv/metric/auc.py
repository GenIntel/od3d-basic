
import torch
import numpy as np


def calc_auc_for_vals_larger_than_thresh(vals, thresh_min, thresh_max, steps=100):
    """
    Calculate the AUC for values larger than a certain threshold.
    
    Args:
        vals (torch.tensor, list or np.array): The input values.
        thresh_min (float): The minimum threshold value.
        thresh_max (float): The maximum threshold value.
        steps (int): The number of steps to use for the AUC calculation.   
    Returns:
        float: The calculated AUC value.
    """

    # first use map values to torch
    if not isinstance(vals, torch.Tensor):
        vals = torch.tensor(vals)
    # create thresholds 
    thresholds = torch.linspace(thresh_min, thresh_max, steps).to(device=vals.device) # T,
    # calculate the AUC using torch and the trapezoidal rule
    auc = (vals[:, None] >= thresholds[None, :]).sum(dim=-1) / steps # N,
    return auc 

def calc_auc_for_vals_smaller_than_thresh(vals, thresh_min, thresh_max, steps=100):
    """
    Calculate the AUC for values smaller than a certain threshold.
    
    Args:
        vals (torch.tensor, list or np.array): The input values.
        thresh_min (float): The minimum threshold value.
        thresh_max (float): The maximum threshold value.
        steps (int): The number of steps to use for the AUC calculation.   
    Returns:
        float: The calculated AUC value.
    """

    # first use map values to torch
    if not isinstance(vals, torch.Tensor):
        vals = torch.tensor(vals)
    # create thresholds 
    thresholds = torch.linspace(thresh_min, thresh_max, steps).to(device=vals.device) # T,
    # calculate the AUC using torch and the trapezoidal rule
    auc = (vals[:, None] <= thresholds[None, :]).sum(dim=-1) / steps # N,
    return auc 