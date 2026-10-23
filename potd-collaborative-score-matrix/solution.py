import numpy as np


def collaborative_scores(U: np.ndarray, V: np.ndarray) -> np.ndarray:

    r_hat=np.dot(U, V.T)
    return r_hat

    # TODO: V.T swaps V's axes to (d, n_items) so the shared dimension d lines up.
    pass
