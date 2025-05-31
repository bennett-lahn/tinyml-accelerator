import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import numpy as np

@cocotb.test()
async def test_dense_layer_256_to_64(dut):
    """Test first dense layer: 256 inputs -> 64 outputs (dense1_relu6)"""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    # Test parameters for first dense layer
    INPUT_SIZE = 256  # Flattened from 4x4x16
    OUTPUT_SIZE = 64  # First dense layer output
    
    # Create test data (smaller range for manageable numbers)
    input_vector = np.random.randint(-10, 10, INPUT_SIZE, dtype=np.int8)
    weight_matrix = np.random.randint(-3, 3, (INPUT_SIZE, OUTPUT_SIZE), dtype=np.int8)
    bias_vector = np.random.randint(-50, 50, OUTPUT_SIZE, dtype=np.int32)
    
    # Load input vector
    for i in range(INPUT_SIZE):
        dut.input_vector[i].value = int(input_vector[i])
    
    # Load weight matrix
    for i in range(INPUT_SIZE):
        for j in range(OUTPUT_SIZE):
            dut.weight_matrix[i][j].value = int(weight_matrix[i, j])
    
    # Load bias vector
    for i in range(OUTPUT_SIZE):
        dut.bias_vector[i].value = int(bias_vector[i])
    
    # Set control signals
    dut.input_valid.value = 1
    dut.start_compute.value = 1
    
    await RisingEdge(dut.clk)
    dut.start_compute.value = 0
    
    # Wait for computation to complete
    cycle_count = 0
    while not dut.computation_complete.value:
        await RisingEdge(dut.clk)
        cycle_count += 1
        if cycle_count > 50000:  # Timeout protection
            assert False, "Computation did not complete within reasonable time"
    
    print(f"Dense 256→64 computation completed in {cycle_count} cycles")
    
    # Read outputs
    hw_output = []
    for i in range(OUTPUT_SIZE):
        hw_output.append(int(dut.output_vector[i].value.signed_integer))
    
    # Calculate expected output using numpy
    expected_output = np.dot(input_vector.astype(np.int32), weight_matrix.astype(np.int32)) + bias_vector
    
    print(f"First 5 hardware outputs: {hw_output[:5]}")
    print(f"First 5 expected outputs: {expected_output[:5]}")
    
    # Verify outputs
    for i in range(OUTPUT_SIZE):
        assert hw_output[i] == expected_output[i], f"Output {i}: got {hw_output[i]}, expected {expected_output[i]}"
    
    print("✓ Dense layer 256→64 test passed!")

@cocotb.test()
async def test_dense_layer_64_to_10(dut):
    """Test second dense layer: 64 inputs -> 10 outputs (output_softmax)"""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    # Test parameters for second dense layer
    INPUT_SIZE = 64   # Output from first dense layer
    OUTPUT_SIZE = 10  # Final classification output
    
    # Create test data
    input_vector = np.random.randint(-20, 20, INPUT_SIZE, dtype=np.int8)
    weight_matrix = np.random.randint(-5, 5, (INPUT_SIZE, OUTPUT_SIZE), dtype=np.int8)
    bias_vector = np.random.randint(-100, 100, OUTPUT_SIZE, dtype=np.int32)
    
    # Load input vector
    for i in range(INPUT_SIZE):
        dut.input_vector[i].value = int(input_vector[i])
    
    # Load weight matrix
    for i in range(INPUT_SIZE):
        for j in range(OUTPUT_SIZE):
            dut.weight_matrix[i][j].value = int(weight_matrix[i, j])
    
    # Load bias vector
    for i in range(OUTPUT_SIZE):
        dut.bias_vector[i].value = int(bias_vector[i])
    
    # Set control signals
    dut.input_valid.value = 1
    dut.start_compute.value = 1
    
    await RisingEdge(dut.clk)
    dut.start_compute.value = 0
    
    # Wait for computation to complete
    cycle_count = 0
    while not dut.computation_complete.value:
        await RisingEdge(dut.clk)
        cycle_count += 1
        if cycle_count > 10000:  # Timeout protection
            assert False, "Computation did not complete within reasonable time"
    
    print(f"Dense 64→10 computation completed in {cycle_count} cycles")
    
    # Read outputs
    hw_output = []
    for i in range(OUTPUT_SIZE):
        hw_output.append(int(dut.output_vector[i].value.signed_integer))
    
    # Calculate expected output using numpy
    expected_output = np.dot(input_vector.astype(np.int32), weight_matrix.astype(np.int32)) + bias_vector
    
    print(f"Hardware outputs: {hw_output}")
    print(f"Expected outputs: {expected_output}")
    
    # Verify outputs
    for i in range(OUTPUT_SIZE):
        assert hw_output[i] == expected_output[i], f"Output {i}: got {hw_output[i]}, expected {expected_output[i]}"
    
    print("✓ Dense layer 64→10 test passed!")

@cocotb.test()
async def test_dense_layer_performance_comparison(dut):
    """Compare performance between different dense layer sizes"""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    test_configs = [
        {"input_size": 64, "output_size": 10, "name": "64→10"},
        {"input_size": 256, "output_size": 64, "name": "256→64"},
        {"input_size": 8, "output_size": 4, "name": "8→4 (small test)"}
    ]
    
    for config in test_configs:
        print(f"\n--- Testing {config['name']} configuration ---")
        
        INPUT_SIZE = config["input_size"]
        OUTPUT_SIZE = config["output_size"]
        
        # Reset
        dut.reset.value = 1
        await RisingEdge(dut.clk)
        await RisingEdge(dut.clk)
        dut.reset.value = 0
        await RisingEdge(dut.clk)
        
        # Create test data
        input_vector = np.random.randint(-10, 10, INPUT_SIZE, dtype=np.int8)
        weight_matrix = np.random.randint(-3, 3, (INPUT_SIZE, OUTPUT_SIZE), dtype=np.int8)
        bias_vector = np.random.randint(-50, 50, OUTPUT_SIZE, dtype=np.int32)
        
        # Load data
        for i in range(INPUT_SIZE):
            dut.input_vector[i].value = int(input_vector[i])
        
        for i in range(INPUT_SIZE):
            for j in range(OUTPUT_SIZE):
                dut.weight_matrix[i][j].value = int(weight_matrix[i, j])
        
        for i in range(OUTPUT_SIZE):
            dut.bias_vector[i].value = int(bias_vector[i])
        
        # Start computation and measure cycles
        dut.input_valid.value = 1
        dut.start_compute.value = 1
        await RisingEdge(dut.clk)
        dut.start_compute.value = 0
        
        cycle_count = 0
        while not dut.computation_complete.value:
            await RisingEdge(dut.clk)
            cycle_count += 1
            if cycle_count > 100000:
                assert False, f"Computation did not complete for {config['name']}"
        
        # Calculate theoretical cycles (INPUT_SIZE * OUTPUT_SIZE + overhead)
        theoretical_cycles = INPUT_SIZE * OUTPUT_SIZE + 2  # +2 for INITIALIZE and transition
        
        print(f"  Actual cycles: {cycle_count}")
        print(f"  Theoretical cycles: {theoretical_cycles}")
        print(f"  Efficiency: {theoretical_cycles/cycle_count*100:.1f}%")
        
        # Verify correctness
        hw_output = []
        for i in range(OUTPUT_SIZE):
            hw_output.append(int(dut.output_vector[i].value.signed_integer))
        
        expected_output = np.dot(input_vector.astype(np.int32), weight_matrix.astype(np.int32)) + bias_vector
        
        for i in range(OUTPUT_SIZE):
            assert hw_output[i] == expected_output[i], f"{config['name']} Output {i}: got {hw_output[i]}, expected {expected_output[i]}"
        
        print(f"  ✓ {config['name']} test passed!")
    
    print("\n✓ All performance comparison tests passed!") 