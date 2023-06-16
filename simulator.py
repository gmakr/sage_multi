import torch
import numpy as np
# Define the two functions
def f1(X):
    return (X[:, 0])**2 + (X[:, 1])**2

def f2(X):
    return (X[:, 0]-2)**2 + (X[:, 1]-2)**2

def benchmark_fun(X):
    # Convert numpy array to torch tensor
    if isinstance(X, np.ndarray):
        X = torch.from_numpy(X)
    # If a single input point was passed, add an extra dimension
    if X.dim() == 1:
        X = X.unsqueeze(0)

    # Calculate function values
    f1_value = f1(X)
    f2_value = f2(X)

    # Return function values
    return f1_value.detach().numpy(), f2_value.detach().numpy()
