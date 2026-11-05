"""Python interface to the accelerated discrete-event kernel."""
import numpy as np
from . import _native

def simulate(config, model_id, descriptors, batch, mode, baseline):
    return _native.simulate(config, config['models'][model_id],
                            np.asarray(descriptors, dtype=np.int32), batch,
                            mode == 'decode', baseline)
