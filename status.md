

07/25 -------------------------------------------------------------------------
Software Development
Small fixes
Time step fix
Same gait fingerprint across models
Get rid of the wrong datapoints on AB10
Establish test cases to prove functional validity
Tuning
Get better Tunes to minimize rmse, estimate small changes quickly in ramp, phase, or other. Expect to fix within a few steps.
Paper
Test Claims- find hypothesis and test in sim


Comments - tunning is doing well

08/01-------------------------------------------------------------------------
Software Development
    1. Fix the calculation of gait fingerprint so it takes into consideration
        the information for all models
        
    2. Get rid of the wrong datapoints for subject AB10 (low priority)
    3. Think of test cases to test functionality from a math perspective
    
Tunning
    1. Mostly Done Tbh
    
Paper
    1. Create test case to prove hypothesis



09/15-------------------------------------------------------------------------

GEM Conference 2021
    1. Created code to generate figures

Model Fitting
    1. Removed deriviatives from kronecker model
        Simplifies class and makes everything just be pandas/numpy which makes it jit-able

09/21-------------------------------------------------------------------------

Updated datafiles to have knee, hip velocity

Created the calculation from thigh angle and velocity