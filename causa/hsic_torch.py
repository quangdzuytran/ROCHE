"""
Hilbert Schmidt Information Criterion with a Gaussian kernel, based on the
following references
[1]: https://link.springer.com/chapter/10.1007/11564089_7
[2]: https://www.researchgate.net/publication/301818817_Kernel-based_Tests_for_Joint_Independence

"""
import torch


def centering(M: torch.Tensor):
    """
    Calculate the centering matrix
    """
    n = M.shape[0]
    device = M.device
    unit = torch.ones([n, n]).double().to(device)
    identity = torch.eye(n).double().to(device)
    H = identity - unit / n

    return torch.matmul(M, H)


def gaussian_grammat(x: torch.Tensor, sigma=None):
    """
    Calculate the Gram matrix of x using a Gaussian kernel.
    If the bandwidth sigma is None, it is estimated using the median heuristic:
    ||x_i - x_j||**2 = 2 sigma**2
    """
    try:
        x.shape[1]
    except IndexError:
        x = x.unsqueeze(-1)

    xxT = torch.matmul(x, x.T)
    xnorm = torch.diag(xxT) - xxT + (torch.diag(xxT) - xxT).T

    if sigma is None:
        mdist = torch.median(xnorm[xnorm != 0])
        sigma = torch.sqrt(mdist * 0.5)
    
    if sigma == 0:
        eps = 7./3 - 4./3 - 1
        sigma += eps

    KX = - 0.5 * xnorm / sigma / sigma
    KX = torch.exp(KX)
    return KX


def HSIC(x: torch.Tensor, y: torch.Tensor):
    """
    Calculate the HSIC estimator for d=2, as in [1] eq (9)
    """
    n = x.shape[0]
    score = torch.trace(torch.matmul(centering(gaussian_grammat(x)), centering(gaussian_grammat(y)))) / n / n
    return score.squeeze().cpu().numpy()
    