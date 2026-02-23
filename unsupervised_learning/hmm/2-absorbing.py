#!/usr/bin/env python3
"""Module for determining if a Markov chain is absorbing."""
import numpy as np


def absorbing(P):
    """
    Determines if a Markov chain is absorbing
    P: square 2D numpy.ndarray of shape (n, n)
    Returns: True if absorbing, False otherwise
    """
    if not isinstance(P, np.ndarray) or len(P.shape) != 2:
        return False
    n1, n2 = P.shape
    if n1 != n2:
        return False

    # Step 1: Identify absorbing states
    absorbing_states = [i for i in range(n1) if P[i, i] == 1]

    if not absorbing_states:
        return False

    # Step 2: Check if every state can reach an absorbing state
    for start in range(n1):
        visited = set()
        stack = [start]
        reachable = False

        while stack:
            state = stack.pop()
            if state in visited:
                continue
            visited.add(state)

            if state in absorbing_states:
                reachable = True
                break

            # Add neighbors with nonzero transition probability
            neighbors = [j for j in range(n1) if P[state, j] > 0]
            stack.extend(neighbors)

        if not reachable:
            return False

    return True
