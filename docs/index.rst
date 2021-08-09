..
   Note: Items in this toctree form the top-level navigation. See `api.rst` for the `autosummary` directive, and for why `api.rst` isn't called directly.

   .. toctree::
   :hidden:

   Home page <self>
   API reference <_autosummary/app>

Welcome to Decoding Thought's documentation!
============================================

.. .. image:: https://readthedocs.org/projects/spiking-brains/badge/?version=latest
..    :target: https://spiking-brains.readthedocs.io/en/latest/?badge=latest
..    :alt: Documentation Status

Decoding visual stimulus from brain fMRI using methods from computational neuroscience and deep learning.

.. .. figure:: ../assets/spiking-brains.png
..    :alt: Spiking Brains


Content
-------

The modules reside in the package ``./src``.

Following are the notebooks that use function from ``./src/`` to perform
analysis.

1. ``Exploratory Analysis``

   -  Experiment flow.

2. ``Deep Learning``

   -  Decoding fMRI activity.

Installation
------------

Clone the repository.
~~~~~~~~~~~~~~~~~~~~~

.. code:: sh

   git clone https://github.com/theairbend3r/decoding-thoughts.git

Install the packages.
~~~~~~~~~~~~~~~~~~~~~

Create environment.

.. code:: sh

   conda env create -f environment.yml

Install packages.

.. code:: sh

   pip install requirements.txt

Meta
----

`Akshaj Verma – @theairbend3r <https://twitter.com/theairbend3r>`_

Distributed under the GNU GPL-V3 license. See ``LICENSE`` for more
information.

https://github.com/theairbend3r/decoding-thoughts

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
