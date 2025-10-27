import torch

"""
The Einstein Sum is a notation defined for Tensor Calculus with the following four rules:

Rule 1: Any twice-repeated index in a single term is summed over.
Rule 2:
    Free Index: Occurs once in the term, cannot be replaced by another free index.
    Dummy Index: Occurs twice in the term, can be replaced with any index except free index.
Rule 3: No index may occur 3 or more times in a given term.
Rule 4: In an equation involving Einstein notation, the free indices on both sides must match.
"""

"""
Examples:
1. Matrix Multiplication
    C[i, j] = sum over k, A[i,k]*B[k,j]

    Einstein Sum:
    C = torch.einsum("ik,kj->ij", A, B)

2. Outer Product
    a.shape = [5,]
    b.shape = [3,]
    outer = torch.einsum("i,j->ij", a, b)

3. Vector Sum
    a.shape = [5]
    sum_a = torch.einsum("i->", a)

4. Can return unsummed axes in any order
    a.shape = [5,4,3]
    transformed_a = torch.einsum("ijk->jik", a)
"""
