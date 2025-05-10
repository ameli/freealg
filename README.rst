.. image:: https://raw.githubusercontent.com/ameli/freeform/refs/heads/main/docs/source/_static/images/icons/logo-freeform-light.png
    :align: left
    :width: 240
    :class: custom-dark

*freeform* is a python package that employs **free** probability for large matrix **form**\ s.

Install
=======

Install with ``pip``:

.. code-block::

    pip install freeform

Alternatively, clone the source code and install with

.. code-block::

    cd source_dir
    pip install .

Documentation
=============

Documentation is available at `ameli.github.io/freeform <https://ameli.github.io/freeform>`__.

Quick Usage
===========

Create and Train a Model
------------------------

.. code-block:: python

    >>> import freeform as ff

Test
====

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
our [Contributing Guidelines](CONTRIBUTING.rst) before submitting their code.
If you do not feel comfortable modifying the code, we also welcome feature
requests and bug reports.

How to Cite
===========

* TBD

  .. code::

      @inproceedings{
          TBD
      }

License
=======

|license|

.. |license| image:: https://img.shields.io/github/license/ameli/freeform
   :target: https://opensource.org/licenses/BSD-3-Clause
