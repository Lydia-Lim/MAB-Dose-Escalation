[![Build Status](https://dev.azure.com/ucbalm/Statistical%20Science%20and%20Innovation/_apis/build/status/Predictive%20Analytics/pa-dose-escalation?branchName=master)](https://dev.azure.com/ucbalm/Statistical%20Science%20and%20Innovation/_build/latest?definitionId=2566&branchName=master)

# Introduction 
This repository contains the work conducted by the Predictive Analytics (PA) team for optimising First-In-Human trials at UCB. The central research aims of the team are:
* To develop representative simulations to test and evaluate methods (whilst remaining accutely aware of  assumptions we may be making, and any constraints which UCB must consider when conducting Phase 1 trials.
* To identify, implement, and validate the current state-of-the-art (SOTA) methods within the pharmaceutical space
* To apply and validate our solutions on real UCB and external trial data
* To deploy our solutions as packages or via an interactive interface (e.g. Dash/RShiny) for trial teams to use.

# Contents
The structure of this repository is as follows:
* `./tests/`: Contains the repository's unit tests.
* `./doseescalation/`: Contains the principle codebase of the repository.
```
pa-dose-escalation
+-- README.md
+-- requirements.md
+-- doseescalation
|   +-- simulated_env.py
+-- tests
|   +-- test_simulated_env.py
```

# Installation (still in development)
Make sure you have the latest version of python installed. To get started, run the following commands:
```
pip install -r requirements.txt
pip install .
...
```

# Examples
TBC...

# References
TBC...
