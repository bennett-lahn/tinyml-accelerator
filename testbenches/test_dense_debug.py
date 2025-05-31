import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
import numpy as np
import os

@cocotb.test()
async def test_simple_2_to_1(dut):
    """Simple test: 2 inputs → 1 output to debug basic operation"""
    
    cocotb.log.info("Testing simple 2→1 dense layer for debugging")
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset.value = 1
    dut.start_compute.value = 0
    dut.input_valid.value = 0
    dut.input_size.value = 0
    dut.output_size.value = 0
    
    # Initialize memory interfaces
    dut.tensor_ram_dout.value = 0
    dut.weight_rom_dout.value = 0
    dut.bias_rom_dout.value = 0
    
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    # Simple test: 2 inputs, 1 output
    # Input: [3, 4], Weight: [2, 1], Bias: 5
    # Expected: 3*2 + 4*1 + 5 = 6 + 4 + 5 = 15
    input_size = 2
    output_size = 1
    
    input_vector = np.array([3, 4], dtype=np.int8)
    weight_matrix = np.array([[2], [1]], dtype=np.int8)  # 2x1 matrix
    bias_vector = np.array([5], dtype=np.int32)
    
    print(f"=== Simple Test: {input_size} → {output_size} ===")
    print(f"Input: {input_vector}")
    print(f"Weight: {weight_matrix.flatten()}")
    print(f"Bias: {bias_vector}")
    print(f"Expected: 3*2 + 4*1 + 5 = {3*2 + 4*1 + 5}")
    
    dut.input_size.value = input_size
    dut.output_size.value = output_size
    dut.input_valid.value = 1
    dut.start_compute.value = 1
    
    await RisingEdge(dut.clk)
    dut.start_compute.value = 0
    
    # Monitor computation with detailed logging
    cycle_count = 0
    max_cycles = 50
    
    while cycle_count < max_cycles:
        await RisingEdge(dut.clk)
        cycle_count += 1
        
        # Get current state for debugging
        current_state = int(dut.current_state.value) if hasattr(dut, 'current_state') else -1
        state_names = ["IDLE", "LOAD_BIAS", "COMPUTE_MAC", "COMPLETE"]
        state_name = state_names[current_state] if 0 <= current_state < len(state_names) else f"UNKNOWN({current_state})"
        
        # Get indices
        input_idx = int(dut.input_idx.value) if hasattr(dut, 'input_idx') else -1
        output_idx = int(dut.output_idx.value) if hasattr(dut, 'output_idx') else -1
        bias_idx = int(dut.bias_idx.value) if hasattr(dut, 'bias_idx') else -1
        
        # Get accumulator
        acc_0 = int(dut.output_accumulator[0].value) if hasattr(dut, 'output_accumulator') else -1
        
        print(f"Cycle {cycle_count:2d}: State={state_name:12s} in_idx={input_idx} out_idx={output_idx} bias_idx={bias_idx} acc[0]={acc_0}")
        
        # Respond to memory requests
        if int(dut.tensor_ram_re.value) == 1:
            addr = int(dut.tensor_ram_addr.value)
            if addr < len(input_vector):
                dut.tensor_ram_dout.value = int(input_vector[addr])
                print(f"          -> tensor_ram[{addr}] = {int(input_vector[addr])}")
                
        if int(dut.weight_rom_re.value) == 1:
            addr = int(dut.weight_rom_addr.value)
            if addr < weight_matrix.size:
                input_idx_calc = addr // output_size
                output_idx_calc = addr % output_size
                if input_idx_calc < weight_matrix.shape[0] and output_idx_calc < weight_matrix.shape[1]:
                    weight_val = int(weight_matrix[input_idx_calc, output_idx_calc])
                    dut.weight_rom_dout.value = weight_val
                    print(f"          -> weight_rom[{addr}] = {weight_val} (in={input_idx_calc}, out={output_idx_calc})")
                else:
                    dut.weight_rom_dout.value = 0
            else:
                dut.weight_rom_dout.value = 0
                
        if int(dut.bias_rom_re.value) == 1:
            addr = int(dut.bias_rom_addr.value)
            if addr < len(bias_vector):
                bias_val = int(bias_vector[addr])
                dut.bias_rom_dout.value = bias_val
                print(f"          -> bias_rom[{addr}] = {bias_val}")
            else:
                dut.bias_rom_dout.value = 0
        
        # Check if computation is complete
        if int(dut.computation_complete.value) == 1:
            print(f"Computation completed at cycle {cycle_count}")
            final_output = int(dut.output_vector[0].value)
            print(f"Final output: {final_output}")
            print(f"Expected: 15")
            print(f"Match: {final_output == 15}")
            break
    
    if cycle_count >= max_cycles:
        print(f"ERROR: Computation did not complete within {max_cycles} cycles")
    
    print("=== End Simple Test ===\n")

@cocotb.test()
async def test_dense_layer_256_to_64(dut):
    """Test dense layer with TPU model sizes: 256 inputs → 64 outputs"""
    
    cocotb.log.info("Testing dense layer with TPU model first dense layer size: 256→64")
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset.value = 1
    dut.start_compute.value = 0
    dut.input_valid.value = 0
    dut.input_size.value = 0
    dut.output_size.value = 0
    
    # Initialize memory interfaces
    dut.tensor_ram_dout.value = 0
    dut.weight_rom_dout.value = 0
    dut.bias_rom_dout.value = 0
    
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    # TPU model first dense layer: 256 inputs, 64 outputs
    input_size = 256
    output_size = 64
    
    # Create test data
    input_vector = np.random.randint(-128, 127, size=input_size, dtype=np.int8)
    weight_matrix = np.random.randint(-128, 127, size=(input_size, output_size), dtype=np.int8)
    bias_vector = np.random.randint(-1000, 1000, size=output_size, dtype=np.int32)
    
    # Calculate expected results
    expected_outputs = []
    for out_idx in range(output_size):
        result = bias_vector[out_idx]
        for in_idx in range(input_size):
            result += int(input_vector[in_idx]) * int(weight_matrix[in_idx, out_idx])
        expected_outputs.append(result)
    
    print(f"=== Testing Dense Layer: {input_size} → {output_size} ===")
    print(f"Input range: [{np.min(input_vector)}, {np.max(input_vector)}]")
    print(f"Weight range: [{np.min(weight_matrix)}, {np.max(weight_matrix)}]")
    print(f"Bias range: [{np.min(bias_vector)}, {np.max(bias_vector)}]")
    print(f"Expected output[0]: {expected_outputs[0]}")
    print(f"Expected output[63]: {expected_outputs[63]}")
    
    dut.input_size.value = input_size
    dut.output_size.value = output_size
    dut.input_valid.value = 1
    dut.start_compute.value = 1
    
    await RisingEdge(dut.clk)
    dut.start_compute.value = 0
    
    # Monitor computation
    cycle_count = 0
    max_cycles = 20000  # Increased for larger computation
    
    while cycle_count < max_cycles:
        await RisingEdge(dut.clk)
        cycle_count += 1
        
        # Respond to memory requests
        if int(dut.tensor_ram_re.value) == 1:
            addr = int(dut.tensor_ram_addr.value)
            if addr < len(input_vector):
                dut.tensor_ram_dout.value = int(input_vector[addr])
                
        if int(dut.weight_rom_re.value) == 1:
            addr = int(dut.weight_rom_addr.value)
            if addr < weight_matrix.size:
                input_idx_calc = addr // output_size
                output_idx_calc = addr % output_size
                if input_idx_calc < weight_matrix.shape[0] and output_idx_calc < weight_matrix.shape[1]:
                    weight_val = int(weight_matrix[input_idx_calc, output_idx_calc])
                    dut.weight_rom_dout.value = weight_val
                else:
                    dut.weight_rom_dout.value = 0
            else:
                dut.weight_rom_dout.value = 0
                
        if int(dut.bias_rom_re.value) == 1:
            addr = int(dut.bias_rom_addr.value)
            if addr < len(bias_vector):
                bias_val = int(bias_vector[addr])
                dut.bias_rom_dout.value = bias_val
            else:
                dut.bias_rom_dout.value = 0
        
        # Check if computation is complete
        if int(dut.computation_complete.value) == 1:
            print(f"Computation completed at cycle {cycle_count}")
            
            # Check first few and last few outputs
            for i in [0, 1, 2, 62, 63]:
                actual_output = int(dut.output_vector[i].value)
                expected_output = expected_outputs[i]
                print(f"Output[{i:2d}]: actual={actual_output:8d}, expected={expected_output:8d}, match={actual_output == expected_output}")
            
            # Verify all outputs are computed (non-zero or matching expected)
            computed_outputs = 0
            for i in range(output_size):
                if int(dut.output_vector[i].value) != 0 or expected_outputs[i] == 0:
                    computed_outputs += 1
            
            print(f"Computed outputs: {computed_outputs}/{output_size}")
            
            # Check that unused outputs remain zero
            unused_outputs_zero = True
            for i in range(output_size, 64):
                if int(dut.output_vector[i].value) != 0:
                    unused_outputs_zero = False
                    print(f"ERROR: Unused output[{i}] = {int(dut.output_vector[i].value)} (should be 0)")
            
            if unused_outputs_zero:
                print("✓ All unused outputs correctly remain zero")
            
            break
    
    if cycle_count >= max_cycles:
        print(f"ERROR: Computation did not complete within {max_cycles} cycles")
    
    print("=== End Test ===\n")

@cocotb.test()
async def test_dense_layer_64_to_10(dut):
    """Test dense layer with TPU model sizes: 64 inputs → 10 outputs"""
    
    cocotb.log.info("Testing dense layer with TPU model second dense layer size: 64→10")
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset.value = 1
    dut.start_compute.value = 0
    dut.input_valid.value = 0
    dut.input_size.value = 0
    dut.output_size.value = 0
    
    # Initialize memory interfaces
    dut.tensor_ram_dout.value = 0
    dut.weight_rom_dout.value = 0
    dut.bias_rom_dout.value = 0
    
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    # TPU model second dense layer: 64 inputs, 10 outputs
    input_size = 64
    output_size = 10
    
    # Create test data
    input_vector = np.random.randint(-128, 127, size=input_size, dtype=np.int8)
    weight_matrix = np.random.randint(-128, 127, size=(input_size, output_size), dtype=np.int8)
    bias_vector = np.random.randint(-1000, 1000, size=output_size, dtype=np.int32)
    
    # Calculate expected results
    expected_outputs = []
    for out_idx in range(output_size):
        result = bias_vector[out_idx]
        for in_idx in range(input_size):
            result += int(input_vector[in_idx]) * int(weight_matrix[in_idx, out_idx])
        expected_outputs.append(result)
    
    print(f"=== Testing Dense Layer: {input_size} → {output_size} ===")
    print(f"Input range: [{np.min(input_vector)}, {np.max(input_vector)}]")
    print(f"Weight range: [{np.min(weight_matrix)}, {np.max(weight_matrix)}]")
    print(f"Bias range: [{np.min(bias_vector)}, {np.max(bias_vector)}]")
    print(f"Expected outputs: {expected_outputs}")
    
    dut.input_size.value = input_size
    dut.output_size.value = output_size
    dut.input_valid.value = 1
    dut.start_compute.value = 1
    
    await RisingEdge(dut.clk)
    dut.start_compute.value = 0
    
    # Monitor computation
    cycle_count = 0
    max_cycles = 5000  # Smaller computation
    
    while cycle_count < max_cycles:
        await RisingEdge(dut.clk)
        cycle_count += 1
        
        # Respond to memory requests
        if int(dut.tensor_ram_re.value) == 1:
            addr = int(dut.tensor_ram_addr.value)
            if addr < len(input_vector):
                dut.tensor_ram_dout.value = int(input_vector[addr])
                
        if int(dut.weight_rom_re.value) == 1:
            addr = int(dut.weight_rom_addr.value)
            if addr < weight_matrix.size:
                input_idx_calc = addr // output_size
                output_idx_calc = addr % output_size
                if input_idx_calc < weight_matrix.shape[0] and output_idx_calc < weight_matrix.shape[1]:
                    weight_val = int(weight_matrix[input_idx_calc, output_idx_calc])
                    dut.weight_rom_dout.value = weight_val
                else:
                    dut.weight_rom_dout.value = 0
            else:
                dut.weight_rom_dout.value = 0
                
        if int(dut.bias_rom_re.value) == 1:
            addr = int(dut.bias_rom_addr.value)
            if addr < len(bias_vector):
                bias_val = int(bias_vector[addr])
                dut.bias_rom_dout.value = bias_val
            else:
                dut.bias_rom_dout.value = 0
        
        # Check if computation is complete
        if int(dut.computation_complete.value) == 1:
            print(f"Computation completed at cycle {cycle_count}")
            
            # Check all outputs
            all_correct = True
            for i in range(output_size):
                actual_output = int(dut.output_vector[i].value)
                expected_output = expected_outputs[i]
                is_correct = actual_output == expected_output
                if not is_correct:
                    all_correct = False
                print(f"Output[{i}]: actual={actual_output:8d}, expected={expected_output:8d}, {'✓' if is_correct else '✗'}")
            
            # Check that unused outputs remain zero
            unused_outputs_zero = True
            for i in range(output_size, 64):
                if int(dut.output_vector[i].value) != 0:
                    unused_outputs_zero = False
                    print(f"ERROR: Unused output[{i}] = {int(dut.output_vector[i].value)} (should be 0)")
            
            if unused_outputs_zero:
                print("✓ All unused outputs correctly remain zero")
            
            if all_correct:
                print("✓ All outputs match expected values")
            else:
                print("✗ Some outputs don't match expected values")
            
            break
    
    if cycle_count >= max_cycles:
        print(f"ERROR: Computation did not complete within {max_cycles} cycles")
    
    print("=== End Test ===")

@cocotb.test()
async def test_3_to_2_debug(dut):
    """Test 3 inputs → 2 outputs to debug multiple output behavior"""
    
    cocotb.log.info("Testing 3→2 dense layer for debugging multiple outputs")
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset.value = 1
    dut.start_compute.value = 0
    dut.input_valid.value = 0
    dut.input_size.value = 0
    dut.output_size.value = 0
    
    # Initialize memory interfaces
    dut.tensor_ram_dout.value = 0
    dut.weight_rom_dout.value = 0
    dut.bias_rom_dout.value = 0
    
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    # Test: 3 inputs, 2 outputs
    # Input: [1, 2, 3]
    # Weight matrix: [[10, 20],    # input 0 weights for output 0, 1
    #                 [30, 40],    # input 1 weights for output 0, 1  
    #                 [50, 60]]    # input 2 weights for output 0, 1
    # Bias: [100, 200]
    # Expected output 0: 1*10 + 2*30 + 3*50 + 100 = 10 + 60 + 150 + 100 = 320
    # Expected output 1: 1*20 + 2*40 + 3*60 + 200 = 20 + 80 + 180 + 200 = 480
    
    input_size = 3
    output_size = 2
    
    input_vector = np.array([1, 2, 3], dtype=np.int8)
    weight_matrix = np.array([[10, 20], [30, 40], [50, 60]], dtype=np.int8)  # 3x2 matrix
    bias_vector = np.array([100, 200], dtype=np.int32)
    
    print(f"=== Test: {input_size} → {output_size} ===")
    print(f"Input: {input_vector}")
    print(f"Weight matrix:\n{weight_matrix}")
    print(f"Bias: {bias_vector}")
    print(f"Expected output 0: 1*10 + 2*30 + 3*50 + 100 = {1*10 + 2*30 + 3*50 + 100}")
    print(f"Expected output 1: 1*20 + 2*40 + 3*60 + 200 = {1*20 + 2*40 + 3*60 + 200}")
    
    dut.input_size.value = input_size
    dut.output_size.value = output_size
    dut.input_valid.value = 1
    dut.start_compute.value = 1
    
    await RisingEdge(dut.clk)
    dut.start_compute.value = 0
    
    # Monitor computation with detailed logging
    cycle_count = 0
    max_cycles = 100
    
    while cycle_count < max_cycles:
        await RisingEdge(dut.clk)
        cycle_count += 1
        
        # Get current state for debugging
        current_state = int(dut.current_state.value) if hasattr(dut, 'current_state') else -1
        state_names = ["IDLE", "LOAD_BIAS", "COMPUTE_MAC", "COMPLETE"]
        state_name = state_names[current_state] if 0 <= current_state < len(state_names) else f"UNKNOWN({current_state})"
        
        # Get indices
        input_idx = int(dut.input_idx.value) if hasattr(dut, 'input_idx') else -1
        output_idx = int(dut.output_idx.value) if hasattr(dut, 'output_idx') else -1
        bias_idx = int(dut.bias_idx.value) if hasattr(dut, 'bias_idx') else -1
        
        # Get accumulators
        acc_0 = int(dut.output_accumulator[0].value) if hasattr(dut, 'output_accumulator') else -1
        acc_1 = int(dut.output_accumulator[1].value) if hasattr(dut, 'output_accumulator') else -1
        
        print(f"Cycle {cycle_count:2d}: State={state_name:12s} in_idx={input_idx} out_idx={output_idx} bias_idx={bias_idx} acc[0]={acc_0} acc[1]={acc_1}")
        
        # Respond to memory requests
        if int(dut.tensor_ram_re.value) == 1:
            addr = int(dut.tensor_ram_addr.value)
            if addr < len(input_vector):
                dut.tensor_ram_dout.value = int(input_vector[addr])
                print(f"          -> tensor_ram[{addr}] = {int(input_vector[addr])}")
                
        if int(dut.weight_rom_re.value) == 1:
            addr = int(dut.weight_rom_addr.value)
            if addr < weight_matrix.size:
                input_idx_calc = addr // output_size
                output_idx_calc = addr % output_size
                if input_idx_calc < weight_matrix.shape[0] and output_idx_calc < weight_matrix.shape[1]:
                    weight_val = int(weight_matrix[input_idx_calc, output_idx_calc])
                    dut.weight_rom_dout.value = weight_val
                    print(f"          -> weight_rom[{addr}] = {weight_val} (in={input_idx_calc}, out={output_idx_calc})")
                else:
                    dut.weight_rom_dout.value = 0
                    print(f"          -> weight_rom[{addr}] = 0 (out of bounds)")
            else:
                dut.weight_rom_dout.value = 0
                print(f"          -> weight_rom[{addr}] = 0 (addr out of bounds)")
                
        if int(dut.bias_rom_re.value) == 1:
            addr = int(dut.bias_rom_addr.value)
            if addr < len(bias_vector):
                bias_val = int(bias_vector[addr])
                dut.bias_rom_dout.value = bias_val
                print(f"          -> bias_rom[{addr}] = {bias_val}")
            else:
                dut.bias_rom_dout.value = 0
                print(f"          -> bias_rom[{addr}] = 0 (out of bounds)")
        
        # Check if computation is complete
        if int(dut.computation_complete.value) == 1:
            print(f"Computation completed at cycle {cycle_count}")
            output_0 = int(dut.output_vector[0].value)
            output_1 = int(dut.output_vector[1].value)
            print(f"Output[0]: actual={output_0}, expected=320, match={output_0 == 320}")
            print(f"Output[1]: actual={output_1}, expected=480, match={output_1 == 480}")
            break
    
    if cycle_count >= max_cycles:
        print(f"ERROR: Computation did not complete within {max_cycles} cycles")
    
    print("=== End Test ===\n")

if __name__ == "__main__":
    print("Dense layer TPU model size testbench") 