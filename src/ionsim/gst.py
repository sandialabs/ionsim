import numpy as np
import pandas as pd 





class GateSetTomography() # or GST() or GST_Base() if we plan to have child classes.
    """ Class for performing quantum gate set tomography with trapped ions or neutral atoms. 

        Member variables include:
            - Basis where the quantum processes (gates), state, and measurement will live. 
            - initial state: rho_0, representing a state prepared natively. 
            - native measurement: M_0, representing a native measurement. 

            - parametrized_gates is a boolean. If True, the user should specify a model 
                for the gates as a function of the parameters. If False, the class assumes 
                the gate is modeled as a generic, dense process matrix.  
            
            - gate_parameters 


    """ 
    # Questions: 
        # Should we allow more than 1 initial native state and 1 native measurement? 
            # GST manual suggests that ususally only 1 is available. 
    


    # Initial Thoughts: 
    # - user constructs the class. 
    # - user either specifies models for gates upon construction or uses something like an "add_gate_model()" function 
    # - user solves for parameters post-construction 


    # member variables: 
    # - basis where the quantum state, process, and measurement lives. 
    # -  
    basis: Basis
    initial_state: State
    native_measurement: Gate # TODO: what object should this be? 

    # Either specify fiducial prep/measure circuits, or user must include in gate set/list. 
    fiducial_circuit_list: list[str] #e.g. ['g1', 'f1,g1,g2']
    gate_set: list[Gate]

    # I think we need a set of experimental data per circuit listed. 
    # - allow for variable number of shots per experiment. 

    parametrized_gates: bool = False 
    gate_parameters: list[NDArray] 
    system_size: int 


    @classmethod
    def solve_for_all_gate_parameters(cls, solver: str)
        gate_parameters = [] 
        for gate in self.gate_set:
            gate_parameters.append(cls.solve_for_gate_parameters(gate, solver))

        return gate_parameters


    @classmethod
    def solve_for_gate_parameters(cls, gate: Gate, solver: str = 'MLE'):
        """ Function to solve for the parametrization values of a particular gate. """
        parameters = np
        return parameters
        


