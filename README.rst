.. figure:: https://raw.githubusercontent.com/ameli/freealg/refs/heads/main/docs/source/_static/images/icons/logo-freealg-light.png
    :align: left
    :width: 240

`Paper <https://arxiv.org/abs/2506.11994>`__ |
`Slides <https://www.dropbox.com/scl/fi/03gjuyz17k9yhsqy0isoz/free_decomporession_slides.pdf?rlkey=8f82mhciyl2ju02l7hv1md5li&st=26xmhjga&dl=0>`__ |
`Docs <https://ameli.github.io/freealg>`__ |
`Live Demo <https://colab.research.google.com/github/ameli/freealg/blob/main/notebooks/quick_start.ipynb>`__

.. `Slides <https://ameli.github.io/freealg/_static/data/slides.pdf>`__ |

*freealg* is a Python package that employs **free** probability to evaluate the
spectral densities of large matrix **form**\ s. The fundamental algorithm
employed by *freealg* is **free decompression**, which extrapolates from the
empirical spectral densities of small  submatrices to infer the eigenspectrum
of extremely large matrices. 

Install
=======

|pypi|

Install with ``pip``:

.. code-block::

    pip install freealg

Alternatively, clone the source code and install with

.. code-block::

    cd source_dir
    pip install .

Documentation
=============

|deploy-docs|

Documentation is available at `ameli.github.io/freealg <https://ameli.github.io/freealg>`__.

Quick Usage
===========

The following code estimates the eigenvalues of a very large Wishart matrix
using a much smaller Wishart matrix.

.. code-block:: python

    >>> import freealg as fa
    >>> mp = fa.distributions.MarchenkoPastur(1/50) # Wishart matrices with aspect ratio 1/50
    >>> A = mp.matrix(1000)                         # Sample a 1000 x 1000 Wishart matrix
    >>> eigs = fa.eigvalsh(A, 100_000)              # Estimate the eigenvalues of 100000 x 100000

For more details on how to interface with *freealg* check out the 
`Live Demo <https://colab.research.google.com/github/ameli/freealg/blob/main/notebooks/quick_start.ipynb>`__.


Test
====

|build-linux| 

You may test the package with `tox <https://tox.wiki/>`__:

.. code-block::

    cd source_dir
    tox

Alternatively, test with `pytest <https://pytest.org>`__:

.. code-block::

    cd source_dir
    pytest

How to Contribute
=================

We welcome contributions via GitHub's pull request. Developers should review
our `Contributing Guidelines <https://github.com/ameli/freealg/blob/main/CONTRIBUTING.rst>`__
before submitting their code. If you do not feel comfortable modifying the
code, we also welcome feature requests and bug reports.

How to Cite
===========

If you use this work, please cite our `paper <https://openreview.net/pdf?id=2CeGVUpOd7>`__.

  .. code::

      @inproceedings{
            AMELI-2025,
            title={Spectral Estimation with Free Decompression},
            author={Siavash Ameli and Chris van der Heide and Liam Hodgkinson and Michael W. Mahoney},
            booktitle={The Thirty-ninth Annual Conference on Neural Information Processing Systems},
            year={2025},
            url={https://openreview.net/forum?id=2CeGVUpOd7}
      }


License
=======

|license|

.. |build-linux| image:: https://img.shields.io/github/actions/workflow/status/ameli/freealg/build-linux.yml
   :target: https://github.com/ameli/freealg/actions?query=workflow%3Abuild-linux 
.. |deploy-docs| image:: https://img.shields.io/github/actions/workflow/status/ameli/freealg/deploy-docs.yml?label=docs
   :target: https://github.com/ameli/freealg/actions?query=workflow%3Adeploy-docs
.. |pypi| image:: https://img.shields.io/pypi/v/freealg
   :target: https://pypi.org/project/freealg/
.. |license| image:: https://img.shields.io/github/license/ameli/freealg
   :target: https://opensource.org/licenses/BSD-3-Clause
