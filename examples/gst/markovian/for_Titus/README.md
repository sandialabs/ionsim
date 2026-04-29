This is an example for generating GST "simulated" experimental data using IonSim methods. 


The example "example_for_Titus.py" shows the following:

1) Specify your gate set 
2) Run the circuit planner to generate the GST circuit instructions 
3) Include a python module of gate simulators (e.g. in gate_models.py) that map an input state to an output state. Specify which gates correspond to which gate models.   
4) Loop through all the circuits in the instructions and simulate each one. Estimate outcome probabilities and record the estimated outcomes. Append a file with the outcome.
5) Repeat  
