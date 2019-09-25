import torch
import torch.nn.functional as F

def calculate_inverse_covariance_matrix(raw_covariance_matrix, offset = 1.0):
    #offset = 1.0
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
    
    #support_unsqueeze = support.reshape(k, n, -1)
    
    inv_covariance_exp = inv_covariance_matrix.expand([-1,embedding_dim])
    inv_covariance_unsqueeze = inv_covariance_exp.reshape(k, n, -1)
    
    #sum1 = (support_unsqueeze * inv_covariance_unsqueeze).sum(dim=1)
    
    S = inv_covariance_unsqueeze.sum(dim=1)
    
    #class_prototypes = sum1/S
        
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