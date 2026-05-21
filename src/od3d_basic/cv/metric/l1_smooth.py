
def get_norm_and_l1_and_l2_and_smooth_from_diff(diff, norm_before_l1_l2=True, calc_smooth=True, beta=1., diff_dim=-1):
    if diff_dim is not None:
        err_norm = diff.norm(dim=diff_dim)
    else:
        err_norm = diff.abs()

    if norm_before_l1_l2 and diff_dim is not None:
        err_l2 = (diff ** 2).sum(dim=diff_dim)
        err_l1 = err_norm
    else:
        err_l2 = diff ** 2
        err_l1 = diff.abs()
    
    err_l1_smooth = get_l1_smooth_from_l1_and_l2(l1_err=err_l1, l2_err=err_l2, beta=beta)

    if (not norm_before_l1_l2) and diff_dim is not None:
        err_l2 = err_l2.sum(dim=diff_dim)
        err_l1 = err_l1.sum(dim=diff_dim)
        err_l1_smooth = err_l1_smooth.sum(dim=diff_dim)
    
    return err_norm, err_l1, err_l2, err_l1_smooth


def get_l1_smooth_from_l1_and_l2(l1_err, l2_err, beta=1):
    l1_smooth_err = (l1_err - 0.5 * beta) * (l1_err >= beta).float() + (0.5 * l2_err / beta) * (l1_err < beta).float()
    return l1_smooth_err