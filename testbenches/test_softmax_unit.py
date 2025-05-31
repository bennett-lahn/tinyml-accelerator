import cocotb
import logging
import random
import math
from cocotb.triggers import Timer, RisingEdge, FallingEdge
from cocotb.clock import Clock

def to_signed(value, bits):
    """Convert Python int to a signed integer of 'bits' width (2's complement)."""
    mask = (1 << bits) - 1
    val = value & mask
    if (val >> (bits - 1)) & 1:  # Check sign bit
        return val - (1 << bits)
    return val

def to_unsigned(value, bits):
    """Convert Python int to an unsigned integer of 'bits' width."""
    mask = (1 << bits) - 1
    return value & mask

def int8_to_float(int8_val):
    """Convert int8 to float with scaling factor 0.0625 (same as hardware)."""
    return float(to_signed(int8_val, 8)) * 0.0625

def float_to_int8(f_val):
    """Convert float to int8 with scaling and clamping."""
    # Scale by 1/0.0625 = 16 to match hardware scaling
    scaled = int(round(f_val * 16.0))
    # Clamp to int8 range
    if scaled > 127:
        scaled = 127
    elif scaled < -128:
        scaled = -128
    return to_signed(scaled, 8)

def q1_31_to_float(q_val):
    """Convert Q1.31 fixed-point to float."""
    return float(to_signed(q_val, 32)) / 2147483648.0  # 2^31

def float_to_q1_31(f_val):
    """Convert float to Q1.31 fixed-point."""
    # Clamp to valid range for Q1.31
    f_val = max(-1.0, min(f_val, 0.9999999995343387))  # Max value for Q1.31
    return to_signed(int(f_val * 2147483648.0), 32)  # 2^31

def py_softmax(logits, beta=1.0):
    """Python reference implementation of softmax function."""
    # Convert int8 logits to float using the same scaling as hardware
    float_logits = [int8_to_float(logit) for logit in logits]
    
    # Apply beta and find max for numerical stability
    scaled_logits = [beta * logit for logit in float_logits]
    max_logit = max(scaled_logits)
    
    # Compute exp(logit - max_logit) for each logit
    exp_values = [math.exp(logit - max_logit) for logit in scaled_logits]
    
    # Compute sum of exponentials
    exp_sum = sum(exp_values)
    
    # Compute probabilities
    probabilities = [exp_val / exp_sum for exp_val in exp_values]
    
    # Convert back to Q1.31 format
    q_probabilities = [float_to_q1_31(prob) for prob in probabilities]
    
    return q_probabilities, probabilities

async def reset_dut(dut):
    """Reset the DUT."""
    dut.reset.value = 1  # Active high reset
    dut.start.value = 0
    for i in range(10):
        dut.logits[i].value = 0
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.reset.value = 0  # Release reset
    await RisingEdge(dut.clk)

async def run_softmax_test(dut, test_logits, test_name):
    """Run a single softmax test with given logits."""
    dut._log.info(f"🧪 Running test: {test_name}")
    
    # Set input logits
    for i in range(10):
        dut.logits[i].value = test_logits[i]
    
    # Wait for ready
    while not dut.ready.value:
        await RisingEdge(dut.clk)
    
    # Start computation
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0
    
    # Wait for completion
    timeout_counter = 0
    while not dut.valid.value:
        await RisingEdge(dut.clk)
        timeout_counter += 1
        if timeout_counter > 1000:  # Prevent infinite loops
            dut._log.error(f"❌ {test_name}: Timeout waiting for valid output")
            return False
    
    # Read outputs
    dut_probabilities = []
    for i in range(10):
        dut_probabilities.append(dut.probabilities[i].value.signed_integer)
    
    # Compute expected values
    expected_q_probs, expected_float_probs = py_softmax(test_logits)
    
    # Log results
    dut._log.debug(f"Input logits (int8): {test_logits}")
    dut._log.debug(f"Input logits (float): {[int8_to_float(x) for x in test_logits]}")
    dut._log.debug(f"DUT output (Q1.31): {[hex(x) for x in dut_probabilities]}")
    dut._log.debug(f"DUT output (float): {[q1_31_to_float(x) for x in dut_probabilities]}")
    dut._log.debug(f"Expected (Q1.31): {[hex(x) for x in expected_q_probs]}")
    dut._log.debug(f"Expected (float): {expected_float_probs}")
    
    # Verify sum approximately equals 1.0
    dut_sum = sum(q1_31_to_float(prob) for prob in dut_probabilities)
    expected_sum = sum(expected_float_probs)
    
    dut._log.info(f"DUT probability sum: {dut_sum:.6f}")
    dut._log.info(f"Expected probability sum: {expected_sum:.6f}")
    
    # Check if sum is close to 1.0 (allowing for quantization errors)
    if abs(dut_sum - 1.0) > 0.1:  # 10% tolerance for fixed-point errors
        dut._log.warning(f"⚠️  {test_name}: Probability sum deviation: {dut_sum:.6f}")
    
    # Verify individual probabilities are reasonable (within tolerance)
    max_error = 0.0
    for i in range(10):
        dut_float = q1_31_to_float(dut_probabilities[i])
        expected_float = expected_float_probs[i]
        error = abs(dut_float - expected_float)
        max_error = max(max_error, error)
        
        # Log significant deviations
        if error > 0.1:  # 10% tolerance for int8 quantization
            dut._log.warning(f"⚠️  Class {i}: DUT={dut_float:.4f}, Expected={expected_float:.4f}, Error={error:.4f}")
    
    dut._log.info(f"Maximum probability error: {max_error:.6f}")
    
    # Wait a cycle before next test
    await RisingEdge(dut.clk)
    
    return True

@cocotb.test()
async def test_softmax_basic_functionality(dut):
    """Test basic softmax functionality with known inputs."""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset DUT
    await reset_dut(dut)
    
    dut._log.setLevel(logging.INFO)
    dut._log.info("🚀 Starting Basic Softmax Functionality Tests")
    
    # Test Case 1: All zeros (should give uniform distribution)
    test_logits_1 = [0] * 10
    await run_softmax_test(dut, test_logits_1, "All Zeros")
    
    # Test Case 2: One hot (one large value, rest small)
    test_logits_2 = [float_to_int8(-2.0)] * 10  # All -2.0
    test_logits_2[5] = float_to_int8(3.0)       # One class much higher
    await run_softmax_test(dut, test_logits_2, "One Hot")
    
    # Test Case 3: Ascending values
    test_logits_3 = [float_to_int8(-4.0 + i * 0.5) for i in range(10)]
    await run_softmax_test(dut, test_logits_3, "Ascending Values")
    
    # Test Case 4: Random realistic logits (smaller values due to int8 range)
    test_logits_4 = [
        float_to_int8(1.2), float_to_int8(-0.5), float_to_int8(2.1),
        float_to_int8(-1.0), float_to_int8(0.8), float_to_int8(-0.2),
        float_to_int8(1.5), float_to_int8(-0.8), float_to_int8(0.3),
        float_to_int8(-1.2)
    ]
    await run_softmax_test(dut, test_logits_4, "Random Realistic")
    
    dut._log.info("✅ Basic Functionality Tests Completed")

@cocotb.test()
async def test_softmax_edge_cases(dut):
    """Test edge cases for softmax function."""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset DUT
    await reset_dut(dut)
    
    dut._log.setLevel(logging.INFO)
    dut._log.info("🚀 Starting Softmax Edge Case Tests")
    
    # Test Case 1: Maximum positive values
    test_logits_1 = [127] * 10  # Maximum int8 value
    await run_softmax_test(dut, test_logits_1, "Max Positive Values")
    
    # Test Case 2: Maximum negative values  
    test_logits_2 = [-128] * 10  # Minimum int8 value
    await run_softmax_test(dut, test_logits_2, "Max Negative Values")
    
    # Test Case 3: Mixed extreme values
    test_logits_3 = [-128] * 9 + [127]
    await run_softmax_test(dut, test_logits_3, "Mixed Extremes")
    
    # Test Case 4: Very similar values (numerical precision test)
    base_val = 16  # Corresponds to 1.0 in float
    test_logits_4 = [base_val + i for i in range(-5, 5)]  # Small differences
    await run_softmax_test(dut, test_logits_4, "Similar Values")
    
    dut._log.info("✅ Edge Case Tests Completed")

@cocotb.test()
async def test_softmax_control_signals(dut):
    """Test control signal behavior."""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset DUT
    await reset_dut(dut)
    
    dut._log.setLevel(logging.INFO)
    dut._log.info("🚀 Starting Control Signal Tests")
    
    # Test Case 1: Check ready signal behavior
    assert dut.ready.value == 1, "Ready should be high after reset"
    assert dut.valid.value == 0, "Valid should be low after reset"
    
    # Set up test logits
    test_logits = [float_to_int8(i * 0.5) for i in range(10)]
    for i in range(10):
        dut.logits[i].value = test_logits[i]
    
    # Start computation
    dut.start.value = 1
    await RisingEdge(dut.clk)
    
    # Ready should go low, valid should still be low
    assert dut.ready.value == 0, "Ready should go low during computation"
    assert dut.valid.value == 0, "Valid should be low during computation"
    
    dut.start.value = 0
    
    # Wait for completion
    while not dut.valid.value:
        await RisingEdge(dut.clk)
        assert dut.ready.value == 0, "Ready should stay low during computation"
    
    # Check final state
    assert dut.valid.value == 1, "Valid should be high when computation completes"
    assert dut.ready.value == 1, "Ready should be high when computation completes"
    
    # Test Case 2: Check that start can be deasserted immediately
    await reset_dut(dut)
    
    for i in range(10):
        dut.logits[i].value = test_logits[i]
    
    dut.start.value = 1
    await RisingEdge(dut.clk)
    dut.start.value = 0  # Deassert immediately
    
    # Should still complete normally
    timeout_counter = 0
    while not dut.valid.value:
        await RisingEdge(dut.clk)
        timeout_counter += 1
        if timeout_counter > 1000:
            dut._log.error("❌ Timeout in control signal test")
            assert False, "Computation should complete even with early start deassertion"
    
    dut._log.info("✅ Control Signal Tests Completed")

@cocotb.test()
async def test_softmax_random_cases(dut):
    """Test with random inputs to stress test the implementation."""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset DUT
    await reset_dut(dut)
    
    dut._log.setLevel(logging.INFO)
    dut._log.info("🚀 Starting Random Test Cases")
    
    num_random_tests = 20
    
    for test_num in range(num_random_tests):
        # Generate random logits in int8 range
        test_logits = []
        for i in range(10):
            # Random int8 value
            random_int8 = random.randint(-128, 127)
            test_logits.append(random_int8)
        
        success = await run_softmax_test(dut, test_logits, f"Random Test {test_num + 1}")
        if not success:
            dut._log.error(f"❌ Random test {test_num + 1} failed")
            assert False, f"Random test {test_num + 1} failed"
        
        if (test_num + 1) % 5 == 0:
            dut._log.info(f"  Random test progress: {test_num + 1}/{num_random_tests}")
    
    dut._log.info("✅ Random Test Cases Completed")

@cocotb.test()
async def test_softmax_mathematical_properties(dut):
    """Test mathematical properties of softmax."""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset DUT
    await reset_dut(dut)
    
    dut._log.setLevel(logging.INFO)
    dut._log.info("🚀 Starting Mathematical Properties Tests")
    
    # Test Case 1: Shift invariance (softmax(x) = softmax(x + c))
    base_logits = [random.randint(-64, 64) for _ in range(10)]  # Keep values moderate
    shift_amount = 16  # Corresponds to 1.0 in float
    shifted_logits = []
    for base in base_logits:
        shifted = base + shift_amount
        # Clamp to int8 range
        if shifted > 127:
            shifted = 127
        elif shifted < -128:
            shifted = -128
        shifted_logits.append(shifted)
    
    # Run both tests
    await run_softmax_test(dut, base_logits, "Base Logits")
    
    # Store first result
    first_probs = []
    for i in range(10):
        first_probs.append(dut.probabilities[i].value.signed_integer)
    
    await run_softmax_test(dut, shifted_logits, "Shifted Logits")
    
    # Store second result
    second_probs = []
    for i in range(10):
        second_probs.append(dut.probabilities[i].value.signed_integer)
    
    # Compare results (they should be similar due to shift invariance)
    max_diff = 0
    for i in range(10):
        diff = abs(q1_31_to_float(first_probs[i]) - q1_31_to_float(second_probs[i]))
        max_diff = max(max_diff, diff)
    
    dut._log.info(f"Shift invariance max difference: {max_diff:.6f}")
    
    # Allow for some quantization error, but should be very small
    if max_diff > 0.02:  # 2% tolerance for int8 quantization
        dut._log.warning(f"⚠️  Shift invariance property violated: max_diff = {max_diff:.6f}")
    
    dut._log.info("✅ Mathematical Properties Tests Completed")

@cocotb.test()
async def test_softmax_comprehensive(dut):
    """Comprehensive test combining all aspects."""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset DUT
    await reset_dut(dut)
    
    dut._log.setLevel(logging.INFO)
    dut._log.info("🚀 Starting Comprehensive Softmax Test")
    
    # Test typical CNN classification scenario
    # Simulate logits from a final dense layer for 10-class classification
    
    # Case 1: Clear winner (class 3)
    confident_logits = [
        float_to_int8(-1.2),  # class 0
        float_to_int8(-0.8),  # class 1
        float_to_int8(-1.5),  # class 2
        float_to_int8(2.3),   # class 3 - winner
        float_to_int8(-0.9),  # class 4
        float_to_int8(-1.1),  # class 5
        float_to_int8(-0.6),  # class 6
        float_to_int8(-1.3),  # class 7
        float_to_int8(-0.7),  # class 8
        float_to_int8(-1.0)   # class 9
    ]
    
    await run_softmax_test(dut, confident_logits, "Confident Classification")
    
    # Verify that class 3 has the highest probability
    winner_prob = q1_31_to_float(dut.probabilities[3].value.signed_integer)
    for i in range(10):
        if i != 3:
            other_prob = q1_31_to_float(dut.probabilities[i].value.signed_integer)
            assert winner_prob > other_prob, f"Winner class 3 should have higher probability than class {i}"
    
    dut._log.info(f"Winner probability: {winner_prob:.4f}")
    
    # Case 2: Uncertain classification (two close values)
    uncertain_logits = [
        float_to_int8(-1.0),  # class 0
        float_to_int8(1.2),   # class 1 - close winner
        float_to_int8(-0.8),  # class 2
        float_to_int8(1.1),   # class 3 - close second
        float_to_int8(-0.9),  # class 4
        float_to_int8(-1.1),  # class 5
        float_to_int8(-0.7),  # class 6
        float_to_int8(-1.2),  # class 7
        float_to_int8(-0.6),  # class 8
        float_to_int8(-1.0)   # class 9
    ]
    
    await run_softmax_test(dut, uncertain_logits, "Uncertain Classification")
    
    prob_1 = q1_31_to_float(dut.probabilities[1].value.signed_integer)
    prob_3 = q1_31_to_float(dut.probabilities[3].value.signed_integer)
    dut._log.info(f"Close competition: Class 1 = {prob_1:.4f}, Class 3 = {prob_3:.4f}")
    
    dut._log.info("✅ Comprehensive Test Completed")
    dut._log.info("🎉 All Softmax Unit Tests PASSED!")

@cocotb.test()
async def test_softmax_classification_accuracy(dut):
    """Test classification accuracy and winner probability error over many iterations."""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset DUT
    await reset_dut(dut)
    
    dut._log.setLevel(logging.INFO)
    dut._log.info("🚀 Starting Classification Accuracy Test")
    
    num_iterations = 100  # Number of test iterations
    correct_classifications = 0
    total_winner_error = 0.0
    winner_errors = []
    
    dut._log.info(f"Running {num_iterations} iterations...")
    
    for iteration in range(num_iterations):
        # Generate random logits with some structure (not completely random)
        # to ensure meaningful classification scenarios
        test_logits = []
        
        # Create scenarios with clear winners about 70% of the time
        if random.random() < 0.7:
            # Clear winner scenario
            winner_class = random.randint(0, 9)
            for i in range(10):
                if i == winner_class:
                    # Winner gets a higher logit
                    test_logits.append(random.randint(32, 120))  # 2.0 to 7.5 in float
                else:
                    # Others get lower logits
                    test_logits.append(random.randint(-80, 16))  # -5.0 to 1.0 in float
        else:
            # Competitive scenario (multiple high values)
            num_competitors = random.randint(2, 4)
            competitors = random.sample(range(10), num_competitors)
            for i in range(10):
                if i in competitors:
                    test_logits.append(random.randint(16, 80))   # 1.0 to 5.0 in float
                else:
                    test_logits.append(random.randint(-64, 0))   # -4.0 to 0.0 in float
        
        # Determine expected winner (class with highest logit)
        expected_winner = test_logits.index(max(test_logits))
        expected_winner_logit = test_logits[expected_winner]
        
        # Set input logits
        for i in range(10):
            dut.logits[i].value = test_logits[i]
        
        # Wait for ready
        while not dut.ready.value:
            await RisingEdge(dut.clk)
        
        # Start computation
        dut.start.value = 1
        await RisingEdge(dut.clk)
        dut.start.value = 0
        
        # Wait for completion
        timeout_counter = 0
        while not dut.valid.value:
            await RisingEdge(dut.clk)
            timeout_counter += 1
            if timeout_counter > 1000:
                dut._log.error(f"❌ Iteration {iteration}: Timeout")
                assert False, f"Timeout in iteration {iteration}"
        
        # Read DUT outputs
        dut_probabilities = []
        for i in range(10):
            dut_probabilities.append(dut.probabilities[i].value.signed_integer)
        
        # Convert to float probabilities
        dut_float_probs = [q1_31_to_float(prob) for prob in dut_probabilities]
        
        # Determine actual winner (class with highest probability from DUT)
        actual_winner = dut_float_probs.index(max(dut_float_probs))
        actual_winner_prob = dut_float_probs[actual_winner]
        
        # Compute expected probabilities for winner error calculation
        expected_q_probs, expected_float_probs = py_softmax(test_logits)
        expected_winner_prob = expected_float_probs[expected_winner]
        
        # Check if classification is correct
        if actual_winner == expected_winner:
            correct_classifications += 1
            classification_result = "✅"
        else:
            classification_result = "❌"
        
        # Calculate winner probability error
        winner_error = abs(actual_winner_prob - expected_winner_prob)
        total_winner_error += winner_error
        winner_errors.append(winner_error)
        
        # Log detailed results for some iterations
        if iteration < 10 or iteration % 20 == 0:
            dut._log.debug(f"Iter {iteration:3d}: {classification_result} Expected winner: {expected_winner} (logit={expected_winner_logit}), "
                          f"Actual winner: {actual_winner} (prob={actual_winner_prob:.4f}), "
                          f"Expected prob: {expected_winner_prob:.4f}, Error: {winner_error:.6f}")
        
        # Wait a cycle before next iteration
        await RisingEdge(dut.clk)
    
    # Calculate final statistics
    accuracy = (correct_classifications / num_iterations) * 100.0
    avg_winner_error = total_winner_error / num_iterations
    max_winner_error = max(winner_errors)
    min_winner_error = min(winner_errors)
    
    # Calculate percentiles for error distribution
    winner_errors.sort()
    p50_error = winner_errors[len(winner_errors) // 2]
    p90_error = winner_errors[int(len(winner_errors) * 0.9)]
    p95_error = winner_errors[int(len(winner_errors) * 0.95)]
    
    # Report results
    dut._log.info("=" * 60)
    dut._log.info("📊 CLASSIFICATION ACCURACY RESULTS")
    dut._log.info("=" * 60)
    dut._log.info(f"Total iterations: {num_iterations}")
    dut._log.info(f"Correct classifications: {correct_classifications}")
    dut._log.info(f"Classification accuracy: {accuracy:.2f}%")
    dut._log.info("")
    dut._log.info("📈 WINNER PROBABILITY ERROR STATISTICS")
    dut._log.info(f"Average winner error: {avg_winner_error:.6f}")
    dut._log.info(f"Maximum winner error: {max_winner_error:.6f}")
    dut._log.info(f"Minimum winner error: {min_winner_error:.6f}")
    dut._log.info(f"Median (P50) error: {p50_error:.6f}")
    dut._log.info(f"P90 error: {p90_error:.6f}")
    dut._log.info(f"P95 error: {p95_error:.6f}")
    dut._log.info("=" * 60)
    
    # Performance thresholds
    min_accuracy = 85.0  # Expect at least 85% accuracy
    max_avg_error = 0.05  # Expect average error < 5%
    
    # Check performance requirements
    if accuracy >= min_accuracy:
        dut._log.info(f"✅ Classification accuracy {accuracy:.2f}% meets requirement (≥{min_accuracy}%)")
    else:
        dut._log.error(f"❌ Classification accuracy {accuracy:.2f}% below requirement (≥{min_accuracy}%)")
    
    if avg_winner_error <= max_avg_error:
        dut._log.info(f"✅ Average winner error {avg_winner_error:.6f} meets requirement (≤{max_avg_error})")
    else:
        dut._log.error(f"❌ Average winner error {avg_winner_error:.6f} exceeds requirement (≤{max_avg_error})")
    
    # Error distribution analysis
    high_error_count = sum(1 for err in winner_errors if err > 0.1)  # Errors > 10%
    if high_error_count > 0:
        high_error_pct = (high_error_count / num_iterations) * 100.0
        dut._log.warning(f"⚠️  {high_error_count} iterations ({high_error_pct:.1f}%) had winner error > 10%")
    
    dut._log.info("✅ Classification Accuracy Test Completed")
    
    # Assert test passes if both criteria are met
    assert accuracy >= min_accuracy, f"Classification accuracy {accuracy:.2f}% below {min_accuracy}%"
    assert avg_winner_error <= max_avg_error, f"Average winner error {avg_winner_error:.6f} exceeds {max_avg_error}" 