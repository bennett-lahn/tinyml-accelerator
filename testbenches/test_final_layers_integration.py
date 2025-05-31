import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import numpy as np
import random

@cocotb.test()
async def test_final_layers_integration(dut):
    """Test the complete final layers pipeline"""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset.value = 1
    dut.start_processing.value = 0
    dut.input_valid.value = 0
    
    # Initialize inputs
    for i in range(64):  # INPUT_CHANNELS
        dut.input_features[i].value = 0
    
    dut.input_row.value = 0
    dut.input_col.value = 0
    
    # Initialize dense weights and bias (simplified for testing)
    for i in range(256):  # FLATTENED_SIZE
        for j in range(10):  # DENSE_OUTPUT_SIZE
            dut.dense_weights[i][j].value = random.randint(-10, 10)
    
    for i in range(10):
        dut.dense_bias[i].value = random.randint(-5, 5)
    
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    
    # Release reset
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    print("=== Testing Final Layers Integration ===")
    
    # Generate test input data (2x2x64 feature map)
    test_features = np.random.randint(-50, 50, size=(2, 2, 64), dtype=np.int8)
    
    print(f"Input feature map shape: {test_features.shape}")
    print(f"Sample input values: {test_features[0, 0, :5]}")
    
    # Start processing
    dut.start_processing.value = 1
    await RisingEdge(dut.clk)
    dut.start_processing.value = 0
    
    # Feed input data (spatial positions)
    for row in range(2):
        for col in range(2):
            dut.input_row.value = row
            dut.input_col.value = col
            dut.input_valid.value = 1
            
            # Set channel data for this spatial position
            for c in range(64):
                dut.input_features[c].value = int(test_features[row, col, c])
            
            await RisingEdge(dut.clk)
            print(f"Fed spatial position ({row}, {col}) with channels: {test_features[row, col, :5]}...")
    
    dut.input_valid.value = 0
    
    # Wait for processing to complete
    print("Waiting for processing to complete...")
    timeout_cycles = 1000
    cycle_count = 0
    
    while not dut.processing_complete.value and cycle_count < timeout_cycles:
        await RisingEdge(dut.clk)
        cycle_count += 1
        
        # Monitor state progression
        if cycle_count % 50 == 0:
            print(f"Cycle {cycle_count}: Still processing...")
    
    if cycle_count >= timeout_cycles:
        print("ERROR: Processing timed out!")
        assert False, "Processing did not complete within timeout"
    
    print(f"Processing completed in {cycle_count} cycles")
    
    # Check outputs
    assert dut.processing_complete.value == 1, "Processing should be complete"
    assert dut.output_valid.value == 1, "Output should be valid"
    
    # Read classification results
    probabilities = []
    for i in range(10):
        prob = int(dut.class_probabilities[i].value)
        probabilities.append(prob)
    
    predicted_class = int(dut.predicted_class.value)
    
    print(f"Class probabilities: {probabilities}")
    print(f"Predicted class: {predicted_class}")
    
    # Verify probabilities are reasonable
    assert 0 <= predicted_class < 10, f"Predicted class {predicted_class} out of range"
    assert all(0 <= p <= 255 for p in probabilities), "Probabilities out of range"
    
    # Verify predicted class has highest probability
    max_prob_class = probabilities.index(max(probabilities))
    assert predicted_class == max_prob_class, f"Predicted class {predicted_class} != max prob class {max_prob_class}"
    
    print("✓ Final layers integration test passed!")

@cocotb.test()
async def test_flatten_stage_detailed(dut):
    """Test the flatten stage in detail"""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    
    print("=== Testing Flatten Stage ===")
    
    # Create known test pattern
    test_pattern = np.arange(256, dtype=np.int8).reshape(2, 2, 64)
    
    # Start processing
    dut.start_processing.value = 1
    await RisingEdge(dut.clk)
    dut.start_processing.value = 0
    
    # Feed input data
    for row in range(2):
        for col in range(2):
            dut.input_row.value = row
            dut.input_col.value = col
            dut.input_valid.value = 1
            
            for c in range(64):
                dut.input_features[c].value = int(test_pattern[row, col, c])
            
            await RisingEdge(dut.clk)
    
    dut.input_valid.value = 0
    
    # Wait for flatten to complete
    while not dut.flatten_inst.flatten_complete.value:
        await RisingEdge(dut.clk)
    
    print("Flatten stage completed")
    
    # Verify flattened vector
    flattened_expected = test_pattern.flatten()
    
    # Check a few sample values
    for i in range(0, 256, 32):  # Sample every 32nd element
        actual = int(dut.flatten_inst.flattened_vector[i].value)
        expected = int(flattened_expected[i])
        print(f"Index {i}: expected {expected}, got {actual}")
        assert actual == expected, f"Mismatch at index {i}"
    
    print("✓ Flatten stage test passed!")

@cocotb.test()
async def test_softmax_functionality(dut):
    """Test softmax functionality with known inputs"""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    
    print("=== Testing Softmax Functionality ===")
    
    # Test with known logits that should produce clear winner
    test_logits = [-10, -5, 20, -8, -12, -15, -3, -7, -9, -11]  # Class 2 should win
    
    # Set up softmax unit directly
    dut.softmax_inst.start_softmax.value = 1
    dut.softmax_inst.input_valid.value = 1
    
    for i in range(10):
        dut.softmax_inst.input_logits[i].value = test_logits[i]
    
    await RisingEdge(dut.clk)
    dut.softmax_inst.start_softmax.value = 0
    dut.softmax_inst.input_valid.value = 0
    
    # Wait for softmax to complete
    while not dut.softmax_inst.softmax_complete.value:
        await RisingEdge(dut.clk)
    
    # Read results
    probabilities = []
    for i in range(10):
        prob = int(dut.softmax_inst.output_probs[i].value)
        probabilities.append(prob)
    
    print(f"Input logits: {test_logits}")
    print(f"Output probabilities: {probabilities}")
    
    # Class 2 should have highest probability
    max_prob_class = probabilities.index(max(probabilities))
    assert max_prob_class == 2, f"Expected class 2 to win, got class {max_prob_class}"
    
    # Probabilities should sum to approximately 255 (due to fixed-point)
    prob_sum = sum(probabilities)
    print(f"Probability sum: {prob_sum}")
    assert 200 <= prob_sum <= 300, f"Probability sum {prob_sum} not reasonable"
    
    print("✓ Softmax functionality test passed!")

@cocotb.test()
async def test_pipeline_stages(dut):
    """Test that pipeline stages execute in correct order"""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    
    print("=== Testing Pipeline Stage Progression ===")
    
    # Initialize with simple data
    for i in range(64):
        dut.input_features[i].value = i % 128
    
    # Start processing
    dut.start_processing.value = 1
    await RisingEdge(dut.clk)
    dut.start_processing.value = 0
    
    # Feed minimal input
    for row in range(2):
        for col in range(2):
            dut.input_row.value = row
            dut.input_col.value = col
            dut.input_valid.value = 1
            await RisingEdge(dut.clk)
    
    dut.input_valid.value = 0
    
    # Monitor state transitions
    states = []
    prev_state = None
    
    for cycle in range(500):
        await RisingEdge(dut.clk)
        
        # Read current state (assuming we can access it)
        current_state = int(dut.current_state.value)
        
        if current_state != prev_state:
            state_names = ["IDLE", "FLATTEN_STAGE", "DENSE_COMPUTE", "RELU_STAGE", "SOFTMAX_STAGE", "COMPLETE"]
            if current_state < len(state_names):
                print(f"Cycle {cycle}: State changed to {state_names[current_state]}")
                states.append(current_state)
            prev_state = current_state
        
        if dut.processing_complete.value:
            break
    
    # Verify state progression
    expected_sequence = [0, 1, 2, 3, 4, 5]  # IDLE -> FLATTEN -> DENSE -> RELU -> SOFTMAX -> COMPLETE
    
    print(f"State sequence: {states}")
    
    # Check that we hit all expected states
    for expected_state in expected_sequence[1:]:  # Skip IDLE
        assert expected_state in states, f"Missing state {expected_state} in sequence"
    
    print("✓ Pipeline stage progression test passed!")

@cocotb.test()
async def test_final_layers_basic(dut):
    """Basic test of final layers integration"""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset.value = 1
    dut.start_processing.value = 0
    dut.input_valid.value = 0
    
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    print("=== Testing Final Layers Integration ===")
    
    # Initialize simple test data
    for i in range(64):
        dut.input_features[i].value = i % 100
    
    # Start processing
    dut.start_processing.value = 1
    await RisingEdge(dut.clk)
    dut.start_processing.value = 0
    
    # Feed input data
    for row in range(2):
        for col in range(2):
            dut.input_row.value = row
            dut.input_col.value = col
            dut.input_valid.value = 1
            await RisingEdge(dut.clk)
    
    dut.input_valid.value = 0
    
    # Wait for completion
    timeout = 1000
    cycles = 0
    
    while not dut.processing_complete.value and cycles < timeout:
        await RisingEdge(dut.clk)
        cycles += 1
    
    print(f"Processing completed in {cycles} cycles")
    assert dut.processing_complete.value == 1
    
    print("✓ Basic final layers test passed!")

if __name__ == "__main__":
    print("Final layers integration tests") 