#!/usr/bin/env python3
'''
    function def forward(
        Observation, Emission, Transition, Initial
    ):
    that performs the forward algorithm for a hidden markov model:
'''


import numpy as np


def forward(Observation, Emission, Transition, Initial):
    '''
    performs the forward algorithm for a hidden markov model
    '''
    # check that Observation is the correct type and dimension
    try:
        # Hidden States
        N = Transition.shape[0]

        # Observations
        T = Observation.shape[0]

        # F == alpha
        # initialization Î±1(j) = Ï€jbj(o1) 1 â‰¤ j â‰¤ N
        F = np.zeros((N, T))
        F[:, 0] = Initial.T * Emission[:, Observation[0]]

        # formula shorturl.at/amtJT
        # Recursion Î±t(j) == âˆ‘Ni=1 Î±tâˆ’1(i)ai jbj(ot); 1â‰¤jâ‰¤N,1<tâ‰¤T
        for t in range(1, T):
            for n in range(N):
                Transitions = Transition[:, n]
                Emissions = Emission[n, Observation[t]]
                F[n, t] = np.sum(Transitions * F[:, t - 1]
                                 * Emissions)

        # Termination P(O|Î») == âˆ‘Ni=1 Î±T (i)
        P = np.sum(F[:, -1])
        return P, F
    except Exception:
        None, None
