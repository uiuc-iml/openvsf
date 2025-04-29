import numpy as np
import torch
import pytest

from vsf.estimator.recursive_optimizer import ObservationLinearization


def make_obs(matrix, var, bias, indices):
    mat = torch.tensor(matrix, dtype=torch.float32)
    v = torch.tensor(var, dtype=torch.float32) if not isinstance(var, (float, int)) else var
    b = torch.tensor(bias, dtype=torch.float32) if bias is not None else None
    idx = torch.tensor(indices, dtype=torch.long) if indices is not None else None
    return ObservationLinearization(matrix=mat, var=v, bias=b, state_indices=idx)

def test_merge_two_disjoint_indices():
    # obs1 sees state [0,2]; obs2 sees state [1,3]
    obs1 = make_obs([[1,0],[0,1]], [0.1,0.2], [0.0,0.0], [0,2])
    obs2 = make_obs([[2,3,4]], 0.3, [1.0], [1,3,4])

    merged: ObservationLinearization = ObservationLinearization.merge(obs1, obs2)

    # merged matrix shape
    assert merged.matrix.shape == (3, 5)

    # first two rows fill cols 0 & 2
    expected1 = torch.zeros(2,5)
    expected1[:,0] = torch.tensor([1.0,0.0])
    expected1[:,2] = torch.tensor([0.0,1.0])
    assert torch.allclose(merged.matrix[:2], expected1)

    # third row fills cols 1,3,4
    expected2 = torch.zeros(1,5)
    expected2[0, [1,3,4]] = torch.tensor([2.0,3.0,4.0])
    assert torch.allclose(merged.matrix[2:], expected2)

    # bias & var concatenation
    assert torch.allclose(merged.bias, torch.tensor([0.0,0.0,1.0]))
    assert torch.allclose(merged.var,  torch.tensor([0.1,0.2,0.3]))

def test_merge_overlapping_indices():
    # obsA sees [0,1]; obsB sees [1,2]
    obsA = make_obs([[1,2]], [0.5], [0.1], [0,1])
    obsB = make_obs([[3,4]], [0.6], [0.2], [1,2])

    merged: ObservationLinearization = ObservationLinearization.merge(obsA, obsB)

    # merged.matrix = [[1,2,0],[0,3,4]]
    assert torch.allclose(merged.matrix, torch.tensor([[1,2,0],[0,3,4]], dtype=torch.float32))
    # biases & vars
    assert torch.allclose(merged.bias, torch.tensor([0.1,0.2]))
    assert torch.allclose(merged.var,  torch.tensor([0.5,0.6]))

def test_merge_full_and_sparse():
    # full-state model on dim=3, then sparse on [2,0]
    full = make_obs([[1,1,1],[2,2,2]], [0.1,0.1], None, None)
    sparse = make_obs([[3,4]], [0.2], [0.3], [2,0])

    merged : ObservationLinearization = ObservationLinearization.merge(full, sparse)

    # merged.matrix[0:2] == full.matrix
    assert torch.allclose(merged.matrix[:2], full.matrix)
    # last row: columns [2,0] get [3,4]
    expected_last = torch.zeros(3)
    expected_last[[2,0]] = torch.tensor([3,4],dtype=torch.float32)
    assert torch.allclose(merged.matrix[2], expected_last)

    # bias: full had None -> zeros; then [0.3]
    assert torch.allclose(merged.bias, torch.tensor([0.0,0.0,0.3]))
    # var: concatenated diag
    assert torch.allclose(merged.var, torch.tensor([0.1,0.1,0.2]))

