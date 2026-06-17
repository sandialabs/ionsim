# Quick Start Guide: GST with Optimized Circuit Planning

## What's New

This example demonstrates the enhanced `GSTCircuitPlanner` with **germ optimization** based on gate models and sensitivity analysis.

## Running the Example

### 1. Navigate to the example directory
```bash
cd /projects/examples/gst/linear_gst
```

### 2. Run the optimization example
```bash
python3 planning_with_optimization.py
```

This will:
- Generate standard GST circuits (baseline)
- Generate optimized GST circuits with germ selection
- Compare the two approaches
- Simulate the optimized circuits

### 3. Expected runtime
- Standard planning: < 1 second
- Optimized planning: ~2-5 seconds (sensitivity analysis)
- Simulation: Depends on number of circuits (default: 100 circuits)

## Key Output Files

| File | Description |
|------|-------------|
| `standard_gst_circuits.gstdata` | Standard circuit plan |
| `standard_gst_circuits_design.yml` | Standard design YAML |
| `optimized_gst_circuits.gstdata` | Optimized circuit plan |
| `optimized_gst_circuits_design.yml` | Optimized design YAML |
| `simulated_optimized_gst_data.gstdata` | Simulated outcomes |

## Analyzing the Results

### Compare with original example
```bash
python3 planning_and_simulation.py
```

This runs the original (non-optimized) workflow for comparison.

### Run GST analysis on optimized data
```bash
# Edit gst_analysis_markovian.py to use optimized files:
# Line 16: fname = './simulated_optimized_gst_data.gstdata'
# Line 23: design_fname = 'optimized_gst_circuits_design.yml'

python3 gst_analysis_markovian.py
```

## Customizing the Example

### Change number of circuits simulated
Edit line 250 in `planning_with_optimization.py`:
```python
simulate_circuits(..., num_circuits=100)  # Change 100 to desired number
```
Set to `None` to simulate all circuits.

### Use different gate models
Edit the `define_gate_models()` function to use your own process matrix functions.

### Adjust germ powers
Change the `germ_powers` parameter in both planners (lines 85 and 105):
```python
germ_powers=[1, 2, 4, 8, 16, 32]  # Add or remove powers as needed
```

## Understanding the Output

### Standard vs Optimized Planning

**Standard Planning:**
- Uses default germ set
- No gate model information
- Fast but generic

**Optimized Planning:**
- Analyzes gate model sensitivity
- Selects most informative germs
- Tailored to your specific gate errors

### Germ Comparison

The script prints which germs are:
- Common to both approaches
- Unique to standard planning
- Unique to optimized planning

Look for germs like `Gxpi2Gxpi2` or `Gypi2Gypi2` that appear in optimized but not standard - these provide additional sensitivity to specific error parameters.

## Troubleshooting

### Syntax errors
```bash
python3 -m py_compile planning_with_optimization.py
```

### Missing dependencies
```bash
pip install -e /projects  # Install IonSim if needed
```

### Slow performance
- Reduce number of candidate germs in `_generate_candidate_germs_1Q()`
- Use simpler gate models for testing
- Simulate fewer circuits initially

## Next Steps

1. **Compare GST results**: Run analysis on both standard and optimized data
2. **Custom gate models**: Replace the simplified models with your actual gate physics
3. **Multi-qubit**: Extend the optimization to 2+ qubit systems (requires implementation)
4. **Experimental integration**: Use with real hardware by replacing the simulators

## More Information

- See `README.md` for detailed documentation
- See `OPTIMIZATION_CHANGES.md` for implementation details
- See `gst_circuit_planner.py` in the source for the core implementation
