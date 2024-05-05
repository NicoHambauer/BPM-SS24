# Business Process Management - Predictive Process Modelling

This repository is part of the lecture Business Process Management at the University of Leipzig by Prof Dr Patrick Zschech.

In this session, to which the material belongs, we talk about Predictive Process Modelling (PPM), also known as Predictive Business Process Monitoring (PBPM), in particular about:

* Exploration of event log data
* Data pre-processing and encoding
* Naive approaches using a Bayesian classifier with a context window of 1 activity. This is a conditional probability approach.
* Simple approaches using Bayes, DT, RF with the entire activity sequence window. This approach involves position coding via one-hot coding.
* Modern approaches using RNNs and LSTMs including the corresponding preprocessing. Here we model the temporal dependency inherently in the model architecture.

For a version of the notebook, which can also be viewed in GitHub, you have to look under the branch **clear-outputs**. There the jupyter notebook outputs are removed and therefore GitHub can display this otherwise very large notebook without any problems.
