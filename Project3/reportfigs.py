from models import *
"""
Optimization and plotting of different models. Could take half an hour
to a few hours to run:
"""
optmodelcomp()


"""
Generate optimal parameter tables for different training fractions. Note
that the Neural network optimization may take many hours, while the xgboost
optimization may take a few hours. To reduce this, the resamps variable
in the xgbtreeopter() and NNopter() functions in models.py may be reduced.
Which gives more inaccurate results, but faster.
These variables are found at lines 1 and 2
"""
trainfraccompdart()
trainfraccompNN()       #do not recommend running this one, as it takes a while
