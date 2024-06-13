import torch
from options import parse_args
from torch import autograd
import numpy as np

args = parse_args()


def get_sigma0(global_epoch, epsilon, delta):
    alpha = np.ceil(np.log2(1 / delta) / epsilon + 1)

    temp = 2 * (epsilon + np.log(delta) / (alpha - 1))
    sigma_0 = np.sqrt(global_epoch * alpha / temp)
    return sigma_0



def compute_fisher_diag(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    fisher_diag = [torch.zeros_like(param) for param in model.parameters()]

    for data, labels in dataloader:
        data, labels = data.to(device), labels.to(device)

        # Calculate output log probabilities
        log_probs = torch.nn.functional.log_softmax(model(data), dim=1)

        for i, label in enumerate(labels):
            log_prob = log_probs[i, label]

            # Calculate first-order derivatives (gradients)
            model.zero_grad()
            grad1 = autograd.grad(log_prob, model.parameters(), create_graph=True, retain_graph=True)

            # Update Fisher diagonal elements
            for fisher_diag_value, grad_value in zip(fisher_diag, grad1):
                fisher_diag_value.add_(grad_value.detach() ** 2)

            # Free up memory by removing computation graph
            del log_prob, grad1

        # Release CUDA memory
        # torch.cuda.empty_cache()

    # Calculate the mean value
    num_samples = len(dataloader.dataset)
    fisher_diag = [fisher_diag_value / num_samples for fisher_diag_value in fisher_diag]

    # Normalize Fisher values layer-wise
    normalized_fisher_diag = []
    for fisher_value in fisher_diag:
        x_min = torch.min(fisher_value)
        x_max = torch.max(fisher_value)
        normalized_fisher_value = (fisher_value - x_min) / (x_max - x_min)
        normalized_fisher_diag.append(normalized_fisher_value)

    return normalized_fisher_diag
