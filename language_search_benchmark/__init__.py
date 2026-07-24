"""CogWorks Week 3 language benchmark: caption-to-image retrieval.

The benchmark asks a student submission to embed pinned caption sets and
pinned ResNet-18 image descriptors with its own trained pipeline, then the
controller (this package, or the portal runner re-using it) computes all
cosine similarities, rankings, and metrics against the COCO gold
caption-to-image map. The scoring path never executes a reference
implementation, so a broken or inert submission lands at chance rather than
at an accidental constant.
"""

__version__ = "0.1.0"
