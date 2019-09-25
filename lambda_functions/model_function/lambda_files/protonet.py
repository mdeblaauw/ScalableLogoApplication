import torch
import torch.nn.functional as F


def calculate_inverse_covariance_matrix(raw_covariance_matrix, offset=1.0):
    # offset = 1.0
    inv_covariance_matrix = offset + F.softplus(raw_covariance_matrix, beta=1, threshold=20)
    return inv_covariance_matrix


def compute_matrix(inv_covariance_matrix: torch.Tensor, k: int, n: int, embedding_dim: int) -> torch.Tensor:
    """Compute class prototypes from support samples.
    # Arguments
        support: torch.Tensor. Tensor of shape (n * k, d) where d is the embedding
            dimension.
        k: int. "k-way" i.e. number of classes in the classification task
        n: int. "n-shot" of the classification task
    # Returns
        class_prototypes: Prototypes aka mean embeddings for each class
    """
    # Reshape so the first dimension indexes by class then take the mean
    # along that dimension to generate the "prototypes" for each class

    # support_unsqueeze = support.reshape(k, n, -1)

    inv_covariance_exp = inv_covariance_matrix.expand([-1, embedding_dim])
    inv_covariance_unsqueeze = inv_covariance_exp.reshape(k, n, -1)

    # sum1 = (support_unsqueeze * inv_covariance_unsqueeze).sum(dim=1)

    S = inv_covariance_unsqueeze.sum(dim=1)

    # class_prototypes = sum1/S

    return S


def compute_prototypes(support: torch.Tensor, k: int, n: int) -> torch.Tensor:
    """Compute class prototypes from support samples.
    # Arguments
        support: torch.Tensor. Tensor of shape (n * k, d) where d is the embedding
            dimension.
        k: int. "k-way" i.e. number of classes in the classification task
        n: int. "n-shot" of the classification task
    # Returns
        class_prototypes: Prototypes aka mean embeddings for each class
    """
    # Reshape so the first dimension indexes by class then take the mean
    # along that dimension to generate the "prototypes" for each class
    class_prototypes = support.reshape(k, n, -1).mean(dim=1)
    return class_prototypes


def pairwise_distances(x: torch.Tensor,
                       y: torch.Tensor,
                       matching_fn: str,
                       S=None) -> torch.Tensor:
    """Efficiently calculate pairwise distances (or other similarity scores) between
    two sets of samples.
    # Arguments
        x: Query samples. A tensor of shape (n_x, d) where d is the embedding dimension
        y: Class prototypes. A tensor of shape (n_y, d) where d is the embedding dimension
        matching_fn: Distance metric/similarity score to compute between samples
    """
    n_x = x.shape[0]
    n_y = y.shape[0]

    if matching_fn == 'l2':
        distances = (
                x.unsqueeze(1).expand(n_x, n_y, -1) -
                y.unsqueeze(0).expand(n_x, n_y, -1)
        ).pow(2).sum(dim=2)
        return distances
    elif matching_fn == 'gaussian':
        distances = ((
                             x.unsqueeze(1).expand(n_x, n_y, -1) -
                             y.unsqueeze(0).expand(n_x, n_y, -1)
                     ).pow(2) * S.unsqueeze(0).expand(n_x, n_y, -1)).sum(dim=2)

        distances = distances.sqrt()
        return distances
    elif matching_fn == 'gaussian_v2':
        difference = x.unsqueeze(1).expand(n_x, n_y, -1) - y.unsqueeze(0).expand(n_x, n_y, -1)
        difference_two = torch.matmul(S.unsqueeze(0).expand(n_x, n_y, -1, -1), difference.unsqueeze(3))
        distances = torch.matmul(difference.unsqueeze(2), difference_two).squeeze()
        distances = distances.sqrt()
        return distances
    elif matching_fn == 'cosine':
        normalised_x = x / (x.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)
        normalised_y = y / (y.pow(2).sum(dim=1, keepdim=True).sqrt() + EPSILON)

        expanded_x = normalised_x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = normalised_y.unsqueeze(0).expand(n_x, n_y, -1)

        cosine_similarities = (expanded_x * expanded_y).sum(dim=2)
        return 1 - cosine_similarities
    elif matching_fn == 'dot':
        expanded_x = x.unsqueeze(1).expand(n_x, n_y, -1)
        expanded_y = y.unsqueeze(0).expand(n_x, n_y, -1)

        return -(expanded_x * expanded_y).sum(dim=2)
    else:
        raise (ValueError('Unsupported similarity function'))