"""GFME — a genomic foundation model ensemble for variant pathogenicity prediction.

The analysis behind the manuscript, from scored data to every table and figure.

    from gfme import config, io
    cfg = config.load()
    X, names, variants = io.load_matrix('clinvar_train', cfg)

Foundation model inference is not part of this package; its outputs are the inputs here. How each
model was scored is described in the supplement of the article, and `data/README.md` records the
provenance of every column.
"""

__version__ = '1.0.0'

from . import config, consequences, features, io, metrics, splits  # noqa: F401

__all__ = ['config', 'consequences', 'features', 'io', 'metrics', 'splits']
