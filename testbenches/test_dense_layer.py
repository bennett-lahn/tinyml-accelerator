import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import numpy as np

@cocotb.test()
async def test_dense_layer_basic(dut):
    """Test basic dense layer computation with small matrices"""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    # Test parameters (small for initial test)
    INPUT_SIZE = 8
    OUTPUT_SIZE = 4
    SYSTOLIC_SIZE = 4
    
    # Create test data
    input_vector = np.random.randint(-128, 127, INPUT_SIZE, dtype=np.int8)
    weight_matrix = np.random.randint(-128, 127, (INPUT_SIZE, OUTPUT_SIZE), dtype=np.int8)
    bias_vector = np.random.randint(-1000, 1000, OUTPUT_SIZE, dtype=np.int32)
    
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
    while not dut.computation_complete.value:
        await RisingEdge(dut.clk)
    
    # Read outputs
    hw_output = []
    for i in range(OUTPUT_SIZE):
        hw_output.append(int(dut.output_vector[i].value.signed_integer))
    
    # Calculate expected output using numpy
    expected_output = np.dot(input_vector.astype(np.int32), weight_matrix.astype(np.int32)) + bias_vector
    
    print(f"Input vector: {input_vector}")
    print(f"Weight matrix shape: {weight_matrix.shape}")
    print(f"Bias vector: {bias_vector}")
    print(f"Hardware output: {hw_output}")
    print(f"Expected output: {expected_output}")
    
    # Verify outputs
    for i in range(OUTPUT_SIZE):
        assert hw_output[i] == expected_output[i], f"Output {i}: got {hw_output[i]}, expected {expected_output[i]}"
    
    print("✓ Dense layer basic test passed!")

@cocotb.test()
async def test_dense_layer_mnist_size(dut):
    """Test dense layer with MNIST-like dimensions (256 -> 10)"""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    # Test parameters (MNIST final layer size)
    INPUT_SIZE = 256  # 4*4*16 flattened
    OUTPUT_SIZE = 10  # 10 classes
    SYSTOLIC_SIZE = 4
    
    # Create test data (smaller range for manageable numbers)
    input_vector = np.random.randint(-10, 10, INPUT_SIZE, dtype=np.int8)
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
    
    # Wait for computation to complete (may take many cycles)
    cycle_count = 0
    while not dut.computation_complete.value:
        await RisingEdge(dut.clk)
        cycle_count += 1
        if cycle_count > 10000:  # Timeout protection
            assert False, "Computation did not complete within reasonable time"
    
    print(f"Computation completed in {cycle_count} cycles")
    
    # Read outputs
    hw_output = []
    for i in range(OUTPUT_SIZE):
        hw_output.append(int(dut.output_vector[i].value.signed_integer))
    
    # Calculate expected output using numpy
    expected_output = np.dot(input_vector.astype(np.int32), weight_matrix.astype(np.int32)) + bias_vector
    
    print(f"Hardware output: {hw_output}")
    print(f"Expected output: {expected_output}")
    
    # Verify outputs
    for i in range(OUTPUT_SIZE):
        assert hw_output[i] == expected_output[i], f"Output {i}: got {hw_output[i]}, expected {expected_output[i]}"
    
    print("✓ Dense layer MNIST size test passed!")

@cocotb.test()
async def test_dense_layer_state_machine(dut):
    """Test the state machine behavior of the dense layer"""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    # Check initial state
    assert dut.computation_complete.value == 0, "Should not be complete initially"
    assert dut.output_valid.value == 0, "Output should not be valid initially"
    
    # Test parameters
    INPUT_SIZE = 8
    OUTPUT_SIZE = 4
    
    # Create simple test data
    input_vector = np.ones(INPUT_SIZE, dtype=np.int8)
    weight_matrix = np.ones((INPUT_SIZE, OUTPUT_SIZE), dtype=np.int8)
    bias_vector = np.zeros(OUTPUT_SIZE, dtype=np.int32)
    
    # Load data
    for i in range(INPUT_SIZE):
        dut.input_vector[i].value = int(input_vector[i])
    
    for i in range(INPUT_SIZE):
        for j in range(OUTPUT_SIZE):
            dut.weight_matrix[i][j].value = int(weight_matrix[i, j])
    
    for i in range(OUTPUT_SIZE):
        dut.bias_vector[i].value = int(bias_vector[i])
    
    # Test without start signal
    dut.input_valid.value = 1
    dut.start_compute.value = 0
    
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    
    assert dut.computation_complete.value == 0, "Should not start without start signal"
    
    # Now start computation
    dut.start_compute.value = 1
    await RisingEdge(dut.clk)
    dut.start_compute.value = 0
    
    # Wait for completion
    while not dut.computation_complete.value:
        await RisingEdge(dut.clk)
    
    # Check outputs (should be INPUT_SIZE for each output since all weights are 1)
    for i in range(OUTPUT_SIZE):
        output_val = int(dut.output_vector[i].value.signed_integer)
        assert output_val == INPUT_SIZE, f"Output {i}: got {output_val}, expected {INPUT_SIZE}"
    
    print("✓ Dense layer state machine test passed!")

@cocotb.test()
async def test_dense_layer_multiple_computations(dut):
    """Test multiple sequential computations"""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    INPUT_SIZE = 8
    OUTPUT_SIZE = 4
    
    # Test multiple different computations
    for test_num in range(3):
        print(f"\n--- Test computation {test_num + 1} ---")
        
        # Create different test data for each iteration
        input_vector = np.random.randint(-10, 10, INPUT_SIZE, dtype=np.int8)
        weight_matrix = np.random.randint(-5, 5, (INPUT_SIZE, OUTPUT_SIZE), dtype=np.int8)
        bias_vector = np.random.randint(-50, 50, OUTPUT_SIZE, dtype=np.int32)
        
        # Load data
        for i in range(INPUT_SIZE):
            dut.input_vector[i].value = int(input_vector[i])
        
        for i in range(INPUT_SIZE):
            for j in range(OUTPUT_SIZE):
                dut.weight_matrix[i][j].value = int(weight_matrix[i, j])
        
        for i in range(OUTPUT_SIZE):
            dut.bias_vector[i].value = int(bias_vector[i])
        
        # Start computation
        dut.input_valid.value = 1
        dut.start_compute.value = 1
        await RisingEdge(dut.clk)
        dut.start_compute.value = 0
        
        # Wait for completion
        while not dut.computation_complete.value:
            await RisingEdge(dut.clk)
        
        # Read and verify outputs
        hw_output = []
        for i in range(OUTPUT_SIZE):
            hw_output.append(int(dut.output_vector[i].value.signed_integer))
        
        expected_output = np.dot(input_vector.astype(np.int32), weight_matrix.astype(np.int32)) + bias_vector
        
        for i in range(OUTPUT_SIZE):
            assert hw_output[i] == expected_output[i], f"Test {test_num}, Output {i}: got {hw_output[i]}, expected {expected_output[i]}"
        
        print(f"Computation {test_num + 1} passed!")
        
        # Wait a few cycles before next computation
        await RisingEdge(dut.clk)
        await RisingEdge(dut.clk)
    
    print("✓ Multiple computations test passed!") 