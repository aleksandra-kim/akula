.. These are examples of badges you might want to add to your README:
   please update the URLs accordingly

    .. image:: https://api.cirrus-ci.com/github/<USER>/akula.svg?branch=main
        :alt: Built Status
        :target: https://cirrus-ci.com/github/<USER>/akula
    .. image:: https://readthedocs.org/projects/akula/badge/?version=latest
        :alt: ReadTheDocs
        :target: https://akula.readthedocs.io/en/stable/
    .. image:: https://img.shields.io/coveralls/github/<USER>/akula/main.svg
        :alt: Coveralls
        :target: https://coveralls.io/r/<USER>/akula
    .. image:: https://img.shields.io/pypi/v/akula.svg
        :alt: PyPI-Server
        :target: https://pypi.org/project/akula/
    .. image:: https://img.shields.io/conda/vn/conda-forge/akula.svg
        :alt: Conda-Forge
        :target: https://anaconda.org/conda-forge/akula
    .. image:: https://pepy.tech/badge/akula/month
        :alt: Monthly Downloads
        :target: https://pepy.tech/project/akula
    .. image:: https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter
        :alt: Twitter
        :target: https://twitter.com/akula

.. image:: https://img.shields.io/badge/-PyScaffold-005CA0?logo=pyscaffold
    :alt: Project generated with PyScaffold
    :target: https://pyscaffold.org/

|

======================================
GSA of Correlated Uncertainties in LCA
======================================

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.12599545.svg
  :alt: Zenodo doi


This GitHub repository provides implementation of the Global Sensitivity Analysis (GSA) protocol for Life Cycle Assessment (LCA) models in the presence of correlated and dependent model inputs. The computations are done for the case study of Swiss household consumption.

This code is part of the publication "Global Sensitivity Analysis of Correlated Uncertainties in Life Cycle Assessment" submitted to the Journal of Industrial Ecology in June 2024. Authors: Aleksandra Kim, Christopher Mutel and Stefanie Hellweg from ETH Zurich and Paul Scherrer Institute in Switzerland.


Abstract
========

Recent advances in research have made global sensitivity analysis of very large and highly linear life cycle assessment systems feasible. In this paper, we build on these developments to include sensitivity analysis of correlated parameters and nonlinear models. We augment numerical uncertainty propagation with Monte Carlo simulations (i) to include propagation of uncertainty from uncertain variables in parameterized inventory datasets; (ii) to account for correlations between process inputs and outputs, and in particular incorporate the carbon balance of combustion activities; (iii) to employ published time-series data instead of static values for electricity generation market mixes in Europe; (iv) to ensure that inputs which are supposed to reach a fixed total (e.g. the percentage contributions of power sources to an electricity mix) actually do so consistently by using the Dirichlet distribution. We then iterate on existing global sensitivity analysis protocols for high-dimensional systems to improve their computational performance. In order to correctly calculate sensitivity rankings for correlated inputs, we use SHapley Additive exPlanations (SHAP) as feature importance metrics with gradient boosted trees. Our results for a case study of climate change impacts of an average Swiss household confirm that neglecting correlations limits the validity of uncertainty and sensitivity analysis. Our methodology and correlated sampling modules are given as open source code.


Installation
============
Create conda environment and install all the necessary packages specified in the requirements.txt file:
::

    conda create --name <environment_name> --file requirements.txt python=3.10 -c conda-forge -c cmutel -c defaults -c anaconda -c haasad
    conda activate <environment_name>
    pip install bw2io==0.9.dev11

Git clone consumption model from https://github.com/aleksandra-kim/consumption_model_ch and do an editable install:
::

   pip install -e .

Install bentso using pip:
::

   pip install bentso


.. _pyscaffold-notes:

Note
====

This project has been set up using PyScaffold 4.1.1. For details and usage
information on PyScaffold see https://pyscaffold.org/.
