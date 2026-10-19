"""Feature transforms used inside saved models.

Kept to numpy only on purpose: a model that contains one of these functions refers to it by
module path when it is pickled, so loading the model imports this module. If it lived next to the
training code it would drag matplotlib, MLflow logging and the database driver into the API.
"""

import numpy as np
import numpy.typing as npt


def signed_log1p(x: npt.ArrayLike) -> npt.NDArray[np.float64]:
    """log(1 + |x|) with the sign kept: compresses heavy-tailed money/count features."""
    values = np.asarray(x, dtype=float)
    return np.sign(values) * np.log1p(np.abs(values))
