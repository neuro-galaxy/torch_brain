.. temporaldata documentation master file, created by
   sphinx-quickstart on Sun Jan  7 15:07:22 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

**temporaldata**
============================

.. image:: https://img.shields.io/pypi/v/temporaldata?color=blue&logo=pypi&logoColor=white
   :target: https://pypi.org/project/temporaldata/
   :alt: PyPI Package

.. image:: https://img.shields.io/badge/GitHub-Repository-black?logo=github&logoColor=white
   :target: https://github.com/neuro-galaxy/temporaldata
   :alt: GitHub Repository

.. image:: https://img.shields.io/badge/GitHub-Issues-black?logo=github&logoColor=white
   :target: https://github.com/neuro-galaxy/temporaldata/issues
   :alt: GitHub Issues


**temporaldata** is a Python package for easily working with temporal data. It provides
advanced data structures and methods to work with multi-modal, multi-resolution 
time series data.

It consists of several key data structures for working with regular timeseries, irregular 
timeseries, and interval objects, and functions for storing, manipulating, and 
accessing temporal data in a flexible but compute and memory-efficient manner. **temporaldata** 
is perfectly suited for deep learning pipelines.



If you encounter any bugs or have feature requests, please submit them to our 
`GitHub Issues page <https://github.com/neuro-galaxy/temporaldata/issues>`_. 

----



.. toctree::
   :maxdepth: 1
   :caption: Get Started

   concepts/installation
   concepts/getting_started

.. toctree::
   :maxdepth: 1
   :caption: Tutorials
   
   concepts/creating_objects
   concepts/data_manipulation
   concepts/interval_operations
   concepts/io

.. toctree::
   :maxdepth: 1
   :caption: Advanced Concepts
   
   concepts/lazy_loading
   concepts/advanced_interval_operations

.. toctree::
   :maxdepth: 2
   :caption: Package Reference

   package/classes/index
   package/functions/index
