import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import numpy as np

@cocotb.test()
async def test_input_driven_dense_layer_both_modes(dut):
    """Test input-driven dense layer with both DENSE1 (256->64) and DENSE2 (64->10) modes"""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    # Test both configurations
    test_configs = [
        {
            "name": "DENSE1 (256→64)",
            "input_size": 256,
            "output_size": 64,
            "input_range": (-10, 10),
            "weight_range": (-3, 3),
            "bias_range": (-50, 50)
        },
        {
            "name": "DENSE2 (64→10)", 
            "input_size": 64,
            "output_size": 10,
            "input_range": (-20, 20),
            "weight_range": (-5, 5),
            "bias_range": (-100, 100)
        }
    ]
    
    for config in test_configs:
        print(f"\n--- Testing {config['name']} ---")
        
        INPUT_SIZE = config["input_size"]
        OUTPUT_SIZE = config["output_size"]
        
        # Set layer configuration through inputs
        dut.input_size.value = INPUT_SIZE
        dut.output_size.value = OUTPUT_SIZE
        await RisingEdge(dut.clk)
        
        # Verify configuration is applied
        assert int(dut.current_input_size.value) == INPUT_SIZE, f"Input size mismatch: got {int(dut.current_input_size.value)}, expected {INPUT_SIZE}"
        assert int(dut.current_output_size.value) == OUTPUT_SIZE, f"Output size mismatch: got {int(dut.current_output_size.value)}, expected {OUTPUT_SIZE}"
        print(f"Configuration verified: {INPUT_SIZE} inputs → {OUTPUT_SIZE} outputs")
        
        # Create test data
        input_vector = np.random.randint(config["input_range"][0], config["input_range"][1], INPUT_SIZE, dtype=np.int8)
        weight_matrix = np.random.randint(config["weight_range"][0], config["weight_range"][1], (INPUT_SIZE, OUTPUT_SIZE), dtype=np.int8)
        bias_vector = np.random.randint(config["bias_range"][0], config["bias_range"][1], OUTPUT_SIZE, dtype=np.int32)
        
        # Load input vector (pad unused entries with zeros)
        for i in range(256):  # Max input size
            if i < INPUT_SIZE:
                dut.input_vector[i].value = int(input_vector[i])
            else:
                dut.input_vector[i].value = 0
        
        # Load weight matrix (pad unused entries with zeros)
        for i in range(256):  # Max input size
            for j in range(64):  # Max output size
                if i < INPUT_SIZE and j < OUTPUT_SIZE:
                    dut.weight_matrix[i][j].value = int(weight_matrix[i, j])
                else:
                    dut.weight_matrix[i][j].value = 0
        
        # Load bias vector (pad unused entries with zeros)
        for i in range(64):  # Max output size
            if i < OUTPUT_SIZE:
                dut.bias_vector[i].value = int(bias_vector[i])
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
            if cycle_count > 50000:
                assert False, f"Computation did not complete for {config['name']}"
        
        print(f"Computation completed in {cycle_count} cycles")
        
        # Read outputs (only the active ones)
        hw_output = []
        for i in range(OUTPUT_SIZE):
            hw_output.append(int(dut.output_vector[i].value.signed_integer))
        
        # Calculate expected output using numpy
        expected_output = np.dot(input_vector.astype(np.int32), weight_matrix.astype(np.int32)) + bias_vector
        
        print(f"First 5 hardware outputs: {hw_output[:5]}")
        print(f"First 5 expected outputs: {expected_output[:5]}")
        
        # Verify outputs
        for i in range(OUTPUT_SIZE):
            assert hw_output[i] == expected_output[i], f"{config['name']} Output {i}: got {hw_output[i]}, expected {expected_output[i]}"
        
        # Verify unused outputs are zero
        for i in range(OUTPUT_SIZE, 64):
            unused_output = int(dut.output_vector[i].value.signed_integer)
            assert unused_output == 0, f"Unused output {i} should be 0, got {unused_output}"
        
        print(f"✓ {config['name']} test passed!")
        
        # Wait a few cycles before next test
        await RisingEdge(dut.clk)
        await RisingEdge(dut.clk)
    
    print("\n✓ All input-driven dense layer tests passed!")

@cocotb.test()
async def test_runtime_reconfiguration(dut):
    """Test switching between configurations at runtime"""
    
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
    
    # Test switching configurations multiple times
    configs = [
        {"input_size": 256, "output_size": 64, "name": "DENSE1 (256→64)"},
        {"input_size": 64, "output_size": 10, "name": "DENSE2 (64→10)"},
        {"input_size": 256, "output_size": 64, "name": "DENSE1 (256→64)"}
    ]
    
    for iteration in range(3):
        for config in configs:
            
            input_size = config["input_size"]
            output_size = config["output_size"]
            layer_name = config["name"]
            
            print(f"\nIteration {iteration+1}, switching to {layer_name}")
            
            # Change configuration
            dut.input_size.value = input_size
            dut.output_size.value = output_size
            await RisingEdge(dut.clk)
            
            # Verify immediate configuration change
            assert int(dut.current_input_size.value) == input_size
            assert int(dut.current_output_size.value) == output_size
            
            # Create simple test data
            input_vector = np.ones(input_size, dtype=np.int8)
            weight_matrix = np.ones((input_size, output_size), dtype=np.int8)
            bias_vector = np.zeros(output_size, dtype=np.int32)
            
            # Load data
            for i in range(256):
                if i < input_size:
                    dut.input_vector[i].value = int(input_vector[i])
                else:
                    dut.input_vector[i].value = 0
            
            for i in range(256):
                for j in range(64):
                    if i < input_size and j < output_size:
                        dut.weight_matrix[i][j].value = int(weight_matrix[i, j])
                    else:
                        dut.weight_matrix[i][j].value = 0
            
            for i in range(64):
                if i < output_size:
                    dut.bias_vector[i].value = int(bias_vector[i])
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
                if cycle_count > 50000:
                    assert False, f"Timeout for {layer_name}"
            
            # Verify outputs (should all be input_size since weights are all 1)
            for i in range(output_size):
                output_val = int(dut.output_vector[i].value.signed_integer)
                assert output_val == input_size, f"{layer_name} Output {i}: got {output_val}, expected {input_size}"
            
            print(f"✓ {layer_name} reconfiguration test passed!")
            
            await RisingEdge(dut.clk)
    
    print("\n✓ Runtime reconfiguration test passed!")

@cocotb.test()
async def test_custom_configurations(dut):
    """Test custom input/output size configurations"""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    print("\n--- Testing Custom Configurations ---")
    
    # Test various custom configurations
    custom_configs = [
        {"input_size": 128, "output_size": 32, "name": "Custom 128→32"},
        {"input_size": 100, "output_size": 20, "name": "Custom 100→20"},
        {"input_size": 50, "output_size": 5, "name": "Custom 50→5"},
        {"input_size": 200, "output_size": 50, "name": "Custom 200→50"}
    ]
    
    for config in custom_configs:
        print(f"\n{config['name']} Test:")
        
        input_size = config["input_size"]
        output_size = config["output_size"]
        
        # Configure layer
        dut.input_size.value = input_size
        dut.output_size.value = output_size
        await RisingEdge(dut.clk)
        
        # Verify configuration
        assert int(dut.current_input_size.value) == input_size
        assert int(dut.current_output_size.value) == output_size
        
        # Create test data
        input_vector = np.random.randint(-5, 5, input_size, dtype=np.int8)
        weight_matrix = np.random.randint(-2, 2, (input_size, output_size), dtype=np.int8)
        bias_vector = np.random.randint(-10, 10, output_size, dtype=np.int32)
        
        # Load data
        for i in range(256):
            if i < input_size:
                dut.input_vector[i].value = int(input_vector[i])
            else:
                dut.input_vector[i].value = 0
        
        for i in range(256):
            for j in range(64):
                if i < input_size and j < output_size:
                    dut.weight_matrix[i][j].value = int(weight_matrix[i, j])
                else:
                    dut.weight_matrix[i][j].value = 0
        
        for i in range(64):
            if i < output_size:
                dut.bias_vector[i].value = int(bias_vector[i])
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
            if cycle_count > 50000:
                assert False, f"Timeout for {config['name']}"
        
        # Verify outputs
        expected_output = np.dot(input_vector.astype(np.int32), weight_matrix.astype(np.int32)) + bias_vector
        for i in range(output_size):
            hw_output = int(dut.output_vector[i].value.signed_integer)
            assert hw_output == expected_output[i], f"Output {i} mismatch: got {hw_output}, expected {expected_output[i]}"
        
        # Verify unused outputs are zero
        for i in range(output_size, 64):
            unused_output = int(dut.output_vector[i].value.signed_integer)
            assert unused_output == 0, f"Unused output {i} should be 0, got {unused_output}"
        
        print(f"  ✓ {config['name']} test passed! ({cycle_count} cycles)")
    
    print("\n✓ Custom configuration tests passed!")

@cocotb.test()
async def test_performance_comparison(dut):
    """Compare performance between different dense layer configurations"""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    print("\n--- Performance Comparison ---")
    
    configs = [
        {"input_size": 256, "output_size": 64, "name": "DENSE1 (256→64)"},
        {"input_size": 64, "output_size": 10, "name": "DENSE2 (64→10)"},
        {"input_size": 128, "output_size": 32, "name": "Custom (128→32)"}
    ]
    
    for config in configs:
        print(f"\n{config['name']} Performance Test:")
        
        input_size = config["input_size"]
        output_size = config["output_size"]
        
        # Configure layer
        dut.input_size.value = input_size
        dut.output_size.value = output_size
        await RisingEdge(dut.clk)
        
        # Create test data
        input_vector = np.random.randint(-5, 5, input_size, dtype=np.int8)
        weight_matrix = np.random.randint(-2, 2, (input_size, output_size), dtype=np.int8)
        bias_vector = np.random.randint(-10, 10, output_size, dtype=np.int32)
        
        # Load data
        for i in range(256):
            if i < input_size:
                dut.input_vector[i].value = int(input_vector[i])
            else:
                dut.input_vector[i].value = 0
        
        for i in range(256):
            for j in range(64):
                if i < input_size and j < output_size:
                    dut.weight_matrix[i][j].value = int(weight_matrix[i, j])
                else:
                    dut.weight_matrix[i][j].value = 0
        
        for i in range(64):
            if i < output_size:
                dut.bias_vector[i].value = int(bias_vector[i])
            else:
                dut.bias_vector[i].value = 0
        
        # Measure performance
        dut.input_valid.value = 1
        dut.start_compute.value = 1
        await RisingEdge(dut.clk)
        dut.start_compute.value = 0
        
        cycle_count = 0
        while not dut.computation_complete.value:
            await RisingEdge(dut.clk)
            cycle_count += 1
        
        # Calculate metrics
        total_operations = input_size * output_size
        theoretical_cycles = total_operations + 2  # +2 for INITIALIZE and transition
        efficiency = (theoretical_cycles / cycle_count) * 100
        
        print(f"  Total operations: {total_operations}")
        print(f"  Actual cycles: {cycle_count}")
        print(f"  Theoretical cycles: {theoretical_cycles}")
        print(f"  Efficiency: {efficiency:.1f}%")
        print(f"  Operations per cycle: {total_operations/cycle_count:.2f}")
        
        # Verify correctness
        expected_output = np.dot(input_vector.astype(np.int32), weight_matrix.astype(np.int32)) + bias_vector
        for i in range(output_size):
            hw_output = int(dut.output_vector[i].value.signed_integer)
            assert hw_output == expected_output[i], f"Output {i} mismatch"
        
        print(f"  ✓ Correctness verified")
    
    print("\n✓ Performance comparison completed!") 