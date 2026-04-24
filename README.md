# Code for PAMPR: Predictability-Aware Motion Planning for Robots

Here is the code used to generate the environment, benchmark the different
planners, and implement the ML+CP system for generating schedules of algorithms
and poses, as well as evaluating against baselines.

## Software requirements


- the Python environment needed can be installed using conda and the yaml file
  provided
- once the environment is activated, install klamp't using pip (further
  instructions at
  https://motion.cs.illinois.edu/software/klampt/latest/pyklampt_docs/Manual-Installation/)
- you'll also need the Klampt and Klampt examples repositories installed from
  https://github.com/krishauser/Klampt and
  https://github.com/krishauser/Klampt-examples
- make a symbolic link called `klampex` to the Klampt-examples directory
  
