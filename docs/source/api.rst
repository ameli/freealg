.. _api:

API Reference
*************

Free Forms
----------

The following classes core implementations defining the free objects.

.. autosummary::
    :toctree: generated
    :caption: Model
    :recursive:
    :template: autosummary/class.rst

    freealg.FreeForm
    freealg.AlgebraicForm
    freealg.GeometricForm

Linear Algebra
--------------

The followings are the free version of some of the common `linalg` functions.

.. autosummary::
    :toctree: generated
    :caption: Linear Algebra
    :recursive:
    :template: autosummary/member.rst

    freealg.eigvalsh
    freealg.cond
    freealg.norm
    freealg.trace
    freealg.slogdet

Distribution Tools
------------------

The following functions are utilities for distributions.

.. autosummary::
    :toctree: generated
    :caption: Distribution Tools
    :recursive:
    :template: autosummary/member.rst

    freealg.supp
    freealg.sample
    freealg.kde

Classical Distributions
-----------------------

The following classes define classical random ensembles.

.. autosummary::
    :toctree: generated
    :caption: Classical Distributions
    :recursive:
    :template: autosummary/class.rst

    freealg.distributions.MarchenkoPastur
    freealg.distributions.Wigner
    freealg.distributions.KestenMcKay
    freealg.distributions.Wachter
    freealg.distributions.Meixner
    freealg.distributions.ChiralBlock
    freealg.distributions.DeformedWigner
    freealg.distributions.DeformedMarchenkoPastur
    freealg.distributions.DeformedJacobi
    freealg.distributions.CompoundPoisson
    freealg.distributions.FussCatalan

Visualization
-------------

Function utilities for visualizing complex functions.

.. autosummary::
    :toctree: generated
    :caption: Visualization
    :recursive:
    :template: autosummary/member.rst

    freealg.visualization.domain_coloring
    freealg.visualization.glue_branches
