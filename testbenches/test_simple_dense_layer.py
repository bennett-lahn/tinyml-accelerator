import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import numpy as np

@cocotb.test()
async def test_simple_dense_layer(dut):
    """Simple test for the input-driven dense layer"""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    print("\n--- Testing Simple Dense Layer (4→2) ---")
    
    # Simple configuration: 4 inputs, 2 outputs
    INPUT_SIZE = 4
    OUTPUT_SIZE = 2
    
    # Set configuration
    dut.input_size.value = INPUT_SIZE
    dut.output_size.value = OUTPUT_SIZE
    await RisingEdge(dut.clk)
    
    # Verify configuration
    assert int(dut.current_input_size.value) == INPUT_SIZE
    assert int(dut.current_output_size.value) == OUTPUT_SIZE
    print(f"Configuration verified: {INPUT_SIZE} inputs → {OUTPUT_SIZE} outputs")
    
    # Simple test data
    input_data = [1, 2, 3, 4]  # 4 inputs
    weights = [
        [1, 2],  # input 0 weights to output 0, 1
        [3, 4],  # input 1 weights to output 0, 1
        [5, 6],  # input 2 weights to output 0, 1
        [7, 8]   # input 3 weights to output 0, 1
    ]
    bias_data = [10, 20]  # bias for output 0, 1
    
    # Load input vector
    for i in range(256):
        if i < INPUT_SIZE:
            dut.input_vector[i].value = input_data[i]
        else:
            dut.input_vector[i].value = 0
    
    # Load weight matrix
    for i in range(256):
        for j in range(64):
            if i < INPUT_SIZE and j < OUTPUT_SIZE:
                dut.weight_matrix[i][j].value = weights[i][j]
            else:
                dut.weight_matrix[i][j].value = 0
    
    # Load bias vector
    for i in range(64):
        if i < OUTPUT_SIZE:
            dut.bias_vector[i].value = bias_data[i]
        else:
            dut.bias_vector[i].value = 0
    
    # Start computation
    dut.input_valid.value = 1
    dut.start_compute.value = 1
    await RisingEdge(dut.clk)
    dut.start_compute.value = 0
    
    # Wait for computation to complete
    cycle_count = 0
    while not dut.computation_complete.value:
        await RisingEdge(dut.clk)
        cycle_count += 1
        if cycle_count > 1000:
            assert False, "Computation did not complete"
    
    print(f"Computation completed in {cycle_count} cycles")
    
    # Read outputs
    hw_output = []
    for i in range(OUTPUT_SIZE):
        output_val = int(dut.output_vector[i].value)
        # Convert from unsigned to signed if needed
        if output_val >= 2**31:
            output_val -= 2**32
        hw_output.append(output_val)
    
    # Calculate expected output manually
    # output[0] = 1*1 + 2*3 + 3*5 + 4*7 + 10 = 1 + 6 + 15 + 28 + 10 = 60
    # output[1] = 1*2 + 2*4 + 3*6 + 4*8 + 20 = 2 + 8 + 18 + 32 + 20 = 80
    expected_output = [60, 80]
    
    print(f"Hardware outputs: {hw_output}")
    print(f"Expected outputs: {expected_output}")
    
    # Verify outputs
    for i in range(OUTPUT_SIZE):
        assert hw_output[i] == expected_output[i], f"Output {i}: got {hw_output[i]}, expected {expected_output[i]}"
    
    # Verify unused outputs are zero
    for i in range(OUTPUT_SIZE, min(10, 64)):  # Check first 10 unused outputs
        unused_output = int(dut.output_vector[i].value)
        if unused_output >= 2**31:
            unused_output -= 2**32
        assert unused_output == 0, f"Unused output {i} should be 0, got {unused_output}"
    
    print("✓ Simple dense layer test passed!")

@cocotb.test()
async def test_medium_dense_layer(dut):
    """Test with medium size configuration"""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    print("\n--- Testing Medium Dense Layer (8→4) ---")
    
    # Medium configuration: 8 inputs, 4 outputs
    INPUT_SIZE = 8
    OUTPUT_SIZE = 4
    
    # Set configuration
    dut.input_size.value = INPUT_SIZE
    dut.output_size.value = OUTPUT_SIZE
    await RisingEdge(dut.clk)
    
    # Simple test data - all ones for easy calculation
    input_data = [1] * INPUT_SIZE
    bias_data = [0] * OUTPUT_SIZE
    
    # Load input vector
    for i in range(256):
        if i < INPUT_SIZE:
            dut.input_vector[i].value = input_data[i]
        else:
            dut.input_vector[i].value = 0
    
    # Load weight matrix - all weights = 1
    for i in range(256):
        for j in range(64):
            if i < INPUT_SIZE and j < OUTPUT_SIZE:
                dut.weight_matrix[i][j].value = 1
            else:
                dut.weight_matrix[i][j].value = 0
    
    # Load bias vector
    for i in range(64):
        if i < OUTPUT_SIZE:
            dut.bias_vector[i].value = bias_data[i]
        else:
            dut.bias_vector[i].value = 0
    
    # Start computation
    dut.input_valid.value = 1
    dut.start_compute.value = 1
    await RisingEdge(dut.clk)
    dut.start_compute.value = 0
    
    # Wait for completion
    cycle_count = 0
    while not dut.computation_complete.value:
        await RisingEdge(dut.clk)
        cycle_count += 1
        if cycle_count > 1000:
            assert False, "Computation did not complete"
    
    print(f"Computation completed in {cycle_count} cycles")
    
    # Read outputs
    hw_output = []
    for i in range(OUTPUT_SIZE):
        output_val = int(dut.output_vector[i].value)
        if output_val >= 2**31:
            output_val -= 2**32
        hw_output.append(output_val)
    
    # Expected: each output should be sum of all inputs = 8
    expected_output = [INPUT_SIZE] * OUTPUT_SIZE
    
    print(f"Hardware outputs: {hw_output}")
    print(f"Expected outputs: {expected_output}")
    
    # Verify outputs
    for i in range(OUTPUT_SIZE):
        assert hw_output[i] == expected_output[i], f"Output {i}: got {hw_output[i]}, expected {expected_output[i]}"
    
    print("✓ Medium dense layer test passed!")

@cocotb.test()
async def test_runtime_reconfiguration_simple(dut):
    """Test runtime reconfiguration with simple cases"""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    print("\n--- Testing Runtime Reconfiguration ---")
    
    configs = [
        {"input_size": 3, "output_size": 2, "name": "3→2"},
        {"input_size": 5, "output_size": 3, "name": "5→3"},
        {"input_size": 2, "output_size": 1, "name": "2→1"}
    ]
    
    for config in configs:
        print(f"\nTesting {config['name']} configuration")
        
        input_size = config["input_size"]
        output_size = config["output_size"]
        
        # Set configuration
        dut.input_size.value = input_size
        dut.output_size.value = output_size
        await RisingEdge(dut.clk)
        
        # Verify configuration
        assert int(dut.current_input_size.value) == input_size
        assert int(dut.current_output_size.value) == output_size
        
        # Simple test: all inputs = 2, all weights = 1, bias = 0
        for i in range(256):
            if i < input_size:
                dut.input_vector[i].value = 2
            else:
                dut.input_vector[i].value = 0
        
        for i in range(256):
            for j in range(64):
                if i < input_size and j < output_size:
                    dut.weight_matrix[i][j].value = 1
                else:
                    dut.weight_matrix[i][j].value = 0
        
        for i in range(64):
            if i < output_size:
                dut.bias_vector[i].value = 0
            else:
                dut.bias_vector[i].value = 0
        
        # Run computation
        dut.input_valid.value = 1
        dut.start_compute.value = 1
        await RisingEdge(dut.clk)
        dut.start_compute.value = 0
        
        # Wait for completion
        cycle_count = 0
        while not dut.computation_complete.value:
            await RisingEdge(dut.clk)
            cycle_count += 1
            if cycle_count > 1000:
                assert False, f"Timeout for {config['name']}"
        
        # Verify outputs (should all be input_size * 2)
        expected_val = input_size * 2
        for i in range(output_size):
            output_val = int(dut.output_vector[i].value)
            if output_val >= 2**31:
                output_val -= 2**32
            assert output_val == expected_val, f"{config['name']} Output {i}: got {output_val}, expected {expected_val}"
        
        print(f"✓ {config['name']} test passed! ({cycle_count} cycles)")
    
    print("\n✓ Runtime reconfiguration test passed!") 