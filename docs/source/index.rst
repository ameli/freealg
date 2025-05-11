.. module:: freeform

|project| Documentation
***********************

.. grid:: 4

    .. grid-item-card:: GitHub
        :link: https://github.com/sameli/freeform
        :text-align: center
        :class-card: custom-card-link

    .. grid-item-card:: PyPI
        :link: https://pypi.org/project/freeform
        :text-align: center
        :class-card: custom-card-link

    .. grid-item-card:: Quick Usage
        :link: quick_usage
        :link-type: ref
        :text-align: center
        :class-card: custom-card-link

    .. grid-item-card:: API reference
        :link: api
        :link-type: ref
        :text-align: center
        :class-card: custom-card-link

Install
=======

Install with ``pip``:

.. prompt:: bash
    
    pip install freeform

Alternatively, clone the source code and install with

.. prompt:: bash
   
    cd source_dir
    pip install .

.. _quick_usage:

Quick Usage
===========


.. code-block:: python

    >>> from freeform import FreeForm

API Reference
=============

Check the list of functions, classes, and modules of |project| with their
usage, options, and examples.

.. toctree::
    :maxdepth: 2
   
    API Reference <api>

Test
====

You may test the package with `tox <https://tox.wiki/>`__:

.. prompt:: bash

    cd source_dir
    tox

Alternatively, test with `pytest <https://pytest.org>`__:

.. prompt:: bash

    cd source_dir
    pytest

How to Contribute
=================

We welcome contributions via GitHub's pull request. Developers should review
our :ref:`Contributing Guidelines <contribute>` before submitting their code.
If you do not feel comfortable modifying the code, we also welcome feature
requests and bug reports.

.. How to Cite
.. include:: cite.rst

License
=======

|license|

.. |license| image:: https://img.shields.io/github/license/ameli/freeform
   :target: https://opensource.org/licenses/BSD-3-Clause
.. |pypi| image:: https://img.shields.io/pypi/v/freeform
.. |traceflows-light| image:: _static/images/icons/logo-freeform-light.svg
   :height: 23
   :class: only-light
.. |traceflows-dark| image:: _static/images/icons/logo-freeform-dark.svg
   :height: 23
   :class: only-dark
