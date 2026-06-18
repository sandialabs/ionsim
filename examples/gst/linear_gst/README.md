# Linear GST Example with Circuit Planning and Germ Optimization

This directory contains examples demonstrating Gate Set Tomography (GST) with IonSim, including both standard and optimized circuit planning.

## Files

### Main Examples

1. **`planning_and_simulation.py`** - Original example showing standard GST circuit planning and simulation
   - Demonstrates basic GST workflow
   - Uses gate-model-agnostic circuit planning
   - Simulates circuits and records outcomes

2. **`planning_with_optimization.py`** - NEW: Enhanced example with optimized planning features
   - Shows both standard and optimized GST circuit planning
   - Demonstrates germ optimization using gate models and sensitivity analysis
   - Compares germ selections between standard and optimized approaches
   - Includes simulation of optimized circuits

### Supporting Files

- **`gate_simulators.py`** - Contains gate simulator functions for Xπ/2, Yπ/2, and idle gates
- **`gst_analysis_markovian.py`** - GST analysis script with Markovian error models
- **`circuit_design.yml`** - Example GST circuit design file
- **`*.gstdata`** - Circuit plan and outcome data files

## New Features Demonstrated

### 1. Standard vs Optimized Planning

The `planning_with_optimization.py` script demonstrates two modes of GST circuit planning:

**Standard Mode** (gate-model-agnostic):
- Uses default germ sequences
- No sensitivity analysis
- Faster but may be less efficient

**Optimized Mode** (gate-model-aware):
- Takes gate model functions as input
- Performs germ sensitivity analysis
- Selects germs that maximize parameter sensitivity
- Can reduce total number of circuits needed

### 2. Germ Optimization

The optimized planner:
1. Generates candidate germ sequences
2. Computes sensitivity of gate parameters to each germ
3. Selects germs that provide best coverage of parameter space
4. Can limit number of germs while maintaining information completeness

### 3. Sensitivity Analysis

The planner includes methods for:
- `compute_germ_sensitivities()` - Computes parameter sensitivity matrices for all gate models in each germ
- `optimize_germs()` - Selects optimal germ set
- `_select_germs_based_on_sensitivity()` - Implements selection strategy

## Running the Examples

### Standard Planning (original)

```bash
python planning_and_simulation.py
```

This will:
1. Generate GST circuits using standard planning
2. Simulate all circuits
3. Write results to `simulated_gst_experimental_data.gstdata`

### Optimized Planning (new)

```bash
python planning_with_optimization.py
```

This will:
1. Run standard planning and save results
2. Run optimized planning with germ selection
3. Compare germ selections
4. Simulate circuits using optimized plan
5. Save all outputs with descriptive filenames

## Key Outputs

When you run `planning_with_optimization.py`, it generates:

1. **Standard planning outputs:**
   - `standard_gst_circuits.gstdata` - Circuit plan
   - `standard_gst_circuits_design.yml` - Design file

2. **Optimized planning outputs:**
   - `optimized_gst_circuits.gstdata` - Optimized circuit plan
   - `optimized_gst_circuits_design.yml` - Optimized design file

3. **Simulation outputs:**
   - `simulated_optimized_gst_data.gstdata` - Simulated measurement outcomes

## Using Optimized Designs with GST Analysis

To analyze the optimized circuits:

1. Update `gst_analysis_markovian.py` to load the optimized design:
   ```python
   design_fname = 'optimized_gst_circuits_design.yml'  # Instead of 'circuit_design.yml'
   gst_circuit_design = sm.GSTCircuitPlanner.load_design(design_fname)
   ```

2. Point to the simulated data:
   ```python
   fname = './simulated_optimized_gst_data.gstdata'
   ```

3. Run the analysis:
   ```bash
   python gst_analysis_markovian.py
   ```

## Customizing Gate Models

To use your own gate models with optimized planning:

1. Define functions that return process matrices:
   ```python
   def my_gate_model(param1, param2, ...):
       # Compute and return d^2 x d^2 process matrix
       return process_matrix
   ```

2. Pass them to the planner:
   ```python
   gate_models = {
       'Gxpi2': my_X_gate_model,
       'Gypi2': my_Y_gate_model,
       'idle': my_idle_model
   }
   
   planner = ism.GSTCircuitPlanner(
       gate_names=gate_names,
       qubit_labels=qubit_labels,
       germ_powers=[1, 2, 4, 8, 16],
       gate_models=gate_models
   )
   ```

## Technical Notes

### Germ Sensitivity

The optimization computes the Frobenius norm of the difference in process matrices when parameters are perturbed:

```
Sensitivity = ||G(θ + ε) - G(θ)||_F / ε
```

Where:
- G is the germ process matrix
- θ are the gate parameters
- ε is a small perturbation

### Selection Strategy

The current implementation selects germs based on:
- Average sensitivity across all parameters and powers
- Cumulative score across all gates
- Configurable number of germs to select

### Performance Considerations

- Optimized planning takes longer than standard planning
- Sensitivity analysis involves matrix exponentiation and finite differences
- Results in more informative circuits, potentially reducing total experiments needed

## Future Enhancements

Potential improvements to the optimization:
- More sophisticated germ selection algorithms
- Adaptive germ power selection
- Multi-qubit germ optimization
- Integration with experimental constraints (e.g., circuit length limits)
