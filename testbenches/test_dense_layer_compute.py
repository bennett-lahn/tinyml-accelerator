import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
import random

@cocotb.test()
async def test_dense_layer_basic_functionality(dut):
    """Test basic dense layer computation with small matrices"""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset.value = 1
    dut.start_compute.value = 0
    dut.input_valid.value = 0
    dut.input_size.value = 3
    dut.output_size.value = 2
    
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    # Test data
    input_vector = [2, 3, 1]  # 3 inputs
    weight_matrix = [
        [1, 2, 3],  # weights for output 0
        [4, 5, 6]   # weights for output 1
    ]
    # Flatten weights in row-major order: [1, 2, 3, 4, 5, 6]
    weights_flat = [1, 2, 3, 4, 5, 6]
    bias_vector = [10, 20]  # 2 biases
    
    # Expected outputs:
    # output[0] = bias[0] + input[0]*weight[0][0] + input[1]*weight[0][1] + input[2]*weight[0][2]
    #           = 10 + 2*1 + 3*2 + 1*3 = 10 + 2 + 6 + 3 = 21
    # output[1] = bias[1] + input[0]*weight[1][0] + input[1]*weight[1][1] + input[2]*weight[1][2]
    #           = 20 + 2*4 + 3*5 + 1*6 = 20 + 8 + 15 + 6 = 49
    expected_outputs = [21, 49]
    
    # Start computation
    dut.input_valid.value = 1
    dut.start_compute.value = 1
    await RisingEdge(dut.clk)
    dut.start_compute.value = 0
    
    # Track outputs as they become ready
    received_outputs = {}
    cycle_count = 0
    
    while cycle_count < 50:  # Safety limit
        await RisingEdge(dut.clk)
        cycle_count += 1
        
        # Print detailed MAC trace for each cycle
        print(f"\n=== Cycle {cycle_count} ===")
        print(f"State: {int(dut.current_state.value) if hasattr(dut, 'current_state') else 'N/A'}")
        print(f"Input idx: {int(dut.input_idx.value) if hasattr(dut, 'input_idx') else 'N/A'}")
        print(f"Output idx: {int(dut.output_idx.value) if hasattr(dut, 'output_idx') else 'N/A'}")
        print(f"Weight idx: {int(dut.weight_idx.value) if hasattr(dut, 'weight_idx') else 'N/A'}")
        
        # MAC unit signals
        if hasattr(dut, 'mac_inst'):
            mac = dut.mac_inst
            print(f"MAC reset: {int(dut.mac_reset.value) if hasattr(dut, 'mac_reset') else 'N/A'}")
            print(f"MAC load_bias: {int(dut.mac_load_bias.value) if hasattr(dut, 'mac_load_bias') else 'N/A'}")
            print(f"MAC bias_in: {int(dut.mac_bias_in.value) if hasattr(dut, 'mac_bias_in') else 'N/A'}")
            print(f"MAC left_in: {int(dut.mac_left_in.value) if hasattr(dut, 'mac_left_in') else 'N/A'}")
            print(f"MAC top_in: {int(dut.mac_top_in.value) if hasattr(dut, 'mac_top_in') else 'N/A'}")
            print(f"MAC sum_out: {int(dut.mac_sum_out.value) if hasattr(dut, 'mac_sum_out') else 'N/A'}")
            
            # MAC internal signals if accessible
            if hasattr(mac, 'mult'):
                print(f"MAC mult: {int(mac.mult.value)}")
            if hasattr(mac, 'sum'):
                print(f"MAC sum: {int(mac.sum.value)}")
            if hasattr(mac, 'accumulator'):
                print(f"MAC accumulator: {int(mac.accumulator.value)}")
        
        # Memory interface signals
        print(f"Tensor RAM RE: {int(dut.tensor_ram_re.value)}")
        print(f"Weight ROM RE: {int(dut.weight_rom_re.value)}")
        print(f"Bias ROM RE: {int(dut.bias_rom_re.value)}")
        
        # Provide memory data based on addresses
        if int(dut.tensor_ram_re.value) == 1:
            addr = int(dut.tensor_ram_addr.value)
            if addr < len(input_vector):
                dut.tensor_ram_dout.value = input_vector[addr]
                print(f"  Tensor RAM read: addr={addr}, data={input_vector[addr]}")
            else:
                dut.tensor_ram_dout.value = 0
                print(f"  Tensor RAM read: addr={addr}, data=0 (out of bounds)")
                
        if int(dut.weight_rom_re.value) == 1:
            addr = int(dut.weight_rom_addr.value)
            if addr < len(weights_flat):
                dut.weight_rom_dout.value = weights_flat[addr]
                print(f"  Weight ROM read: addr={addr}, data={weights_flat[addr]}")
            else:
                dut.weight_rom_dout.value = 0
                print(f"  Weight ROM read: addr={addr}, data=0 (out of bounds)")
                
        if int(dut.bias_rom_re.value) == 1:
            addr = int(dut.bias_rom_addr.value)
            if addr < len(bias_vector):
                dut.bias_rom_dout.value = bias_vector[addr]
                print(f"  Bias ROM read: addr={addr}, data={bias_vector[addr]}")
            else:
                dut.bias_rom_dout.value = 0
                print(f"  Bias ROM read: addr={addr}, data=0 (out of bounds)")
        
        # Output signals
        print(f"Output ready: {int(dut.output_ready.value)}")
        print(f"Output data: {int(dut.output_data.value)}")
        print(f"Output channel: {int(dut.output_channel.value)}")
        print(f"Computation complete: {int(dut.computation_complete.value)}")
        
        # Check if output is ready
        if int(dut.output_ready.value) == 1:
            channel = int(dut.output_channel.value)
            output_data = int(dut.output_data.value)
            received_outputs[channel] = output_data
            print(f"*** Received output for channel {channel}: {output_data} ***")
        
        # Check if computation is complete
        if int(dut.computation_complete.value) == 1:
            print(f"*** Computation completed at cycle {cycle_count} ***")
            break
    
    # Verify outputs
    assert int(dut.computation_complete.value) == 1, "Computation should be complete"
    
    # Check that we received all expected outputs
    assert len(received_outputs) == len(expected_outputs), f"Expected {len(expected_outputs)} outputs, got {len(received_outputs)}"
    
    # Check output values
    for i in range(len(expected_outputs)):
        assert i in received_outputs, f"Missing output for channel {i}"
        actual_output = received_outputs[i]
        expected_output = expected_outputs[i]
        print(f"Output {i}: Expected {expected_output}, Got {actual_output}")
        assert actual_output == expected_output, f"Output {i} mismatch: expected {expected_output}, got {actual_output}"
    
    # Verify configuration echo
    assert int(dut.current_input_size.value) == 3, "Input size should be echoed correctly"
    assert int(dut.current_output_size.value) == 2, "Output size should be echoed correctly"
    
    print("✅ Basic dense layer computation test passed!")

@cocotb.test()
async def test_dense_layer_state_machine(dut):
    """Test the state machine transitions"""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset.value = 1
    dut.start_compute.value = 0
    dut.input_valid.value = 0
    dut.input_size.value = 2
    dut.output_size.value = 2
    
    # Initialize memory interfaces
    dut.tensor_ram_dout.value = 1
    dut.weight_rom_dout.value = 1
    dut.bias_rom_dout.value = 100
    
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    # Should be in IDLE state (0)
    if hasattr(dut, 'current_state'):
        assert int(dut.current_state.value) == 0, "Should start in IDLE state"
    
    # Start computation
    dut.input_valid.value = 1
    dut.start_compute.value = 1
    await RisingEdge(dut.clk)
    dut.start_compute.value = 0
    
    # Should transition to LOAD_BIAS state (1)
    await RisingEdge(dut.clk)
    if hasattr(dut, 'current_state'):
        assert int(dut.current_state.value) == 1, "Should be in LOAD_BIAS state"
    
    # Should transition to COMPUTE_MAC state (2)
    await RisingEdge(dut.clk)
    if hasattr(dut, 'current_state'):
        assert int(dut.current_state.value) == 2, "Should transition to COMPUTE_MAC state"
    
    # Wait for computation to complete
    compute_cycles = 0
    while compute_cycles < 20:  # Safety limit
        await RisingEdge(dut.clk)
        compute_cycles += 1
        if int(dut.computation_complete.value) == 1:
            break
    
    # Should be in COMPLETE state (4) - updated for new state machine
    if hasattr(dut, 'current_state'):
        assert int(dut.current_state.value) == 4, "Should be in COMPLETE state"
    
    print("✅ State machine test passed!")

@cocotb.test()
async def test_dense_layer_memory_addressing(dut):
    """Test memory address generation"""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset.value = 1
    dut.start_compute.value = 0
    dut.input_valid.value = 0
    dut.input_size.value = 3
    dut.output_size.value = 2
    
    # Initialize memory interfaces
    dut.tensor_ram_dout.value = 1
    dut.weight_rom_dout.value = 1
    dut.bias_rom_dout.value = 100
    
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    # Start computation
    dut.input_valid.value = 1
    dut.start_compute.value = 1
    await RisingEdge(dut.clk)
    dut.start_compute.value = 0
    
    # Track memory accesses
    tensor_addresses = []
    weight_addresses = []
    bias_addresses = []
    
    cycle_count = 0
    while cycle_count < 50:
        await RisingEdge(dut.clk)
        cycle_count += 1
        
        # Record memory accesses
        if int(dut.tensor_ram_re.value) == 1:
            tensor_addresses.append(int(dut.tensor_ram_addr.value))
            
        if int(dut.weight_rom_re.value) == 1:
            weight_addresses.append(int(dut.weight_rom_addr.value))
            
        if int(dut.bias_rom_re.value) == 1:
            bias_addresses.append(int(dut.bias_rom_addr.value))
        
        if int(dut.computation_complete.value) == 1:
            break
    
    # Verify bias addresses (should be 0, 1 for 2 outputs)
    expected_bias_addrs = [0, 1]
    assert bias_addresses == expected_bias_addrs, f"Bias addresses: expected {expected_bias_addrs}, got {bias_addresses}"
    
    # Verify tensor RAM addresses (should access inputs 0, 1, 2 multiple times)
    expected_tensor_addrs = [0, 1, 2] * 2  # 3 inputs × 2 outputs
    assert set(tensor_addresses) == set([0, 1, 2]), f"Tensor addresses should include all inputs"
    
    # Verify weight ROM addresses - continuous addressing
    # For 3 inputs × 2 outputs, addresses should be: 0, 1, 2, 3, 4, 5
    expected_weight_addrs = [0, 1, 2, 3, 4, 5]
    assert set(weight_addresses) == set(expected_weight_addrs), f"Weight addresses: expected {expected_weight_addrs}, got unique {list(set(weight_addresses))}"
    
    print("✅ Memory addressing test passed!")

@cocotb.test()
async def test_dense_layer_edge_cases(dut):
    """Test edge cases like single input/output"""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset.value = 1
    dut.start_compute.value = 0
    dut.input_valid.value = 0
    
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    # Test 1: Single input, single output
    dut.input_size.value = 1
    dut.output_size.value = 1
    dut.input_valid.value = 1
    dut.start_compute.value = 1
    
    # Test data: input=5, weight=3, bias=2 -> output=5*3+2=17
    dut.tensor_ram_dout.value = 5
    dut.weight_rom_dout.value = 3
    dut.bias_rom_dout.value = 2
    
    await RisingEdge(dut.clk)
    dut.start_compute.value = 0
    
    # Wait for completion and capture output
    cycle_count = 0
    output_received = False
    actual_output = 0
    
    while cycle_count < 20:
        await RisingEdge(dut.clk)
        cycle_count += 1
        
        # Check if output is ready
        if int(dut.output_ready.value) == 1:
            actual_output = int(dut.output_data.value)
            output_received = True
            print(f"Single output test: Got {actual_output}")
        
        if int(dut.computation_complete.value) == 1:
            break
    
    # Verify single output
    assert output_received, "Should have received output"
    assert actual_output == 17, f"Expected 17, got {actual_output}"
    
    print("✅ Edge cases test passed!")

@cocotb.test()
async def test_dense_layer_multiple_outputs_timing(dut):
    """Test that outputs are produced in correct order and timing"""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset.value = 1
    dut.start_compute.value = 0
    dut.input_valid.value = 0
    dut.input_size.value = 2
    dut.output_size.value = 3
    
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    # Test data
    input_vector = [1, 2]
    weights_flat = [1, 2, 3, 4, 5, 6]  # 2 inputs × 3 outputs
    bias_vector = [10, 20, 30]
    
    # Start computation
    dut.input_valid.value = 1
    dut.start_compute.value = 1
    await RisingEdge(dut.clk)
    dut.start_compute.value = 0
    
    # Track output order
    output_order = []
    cycle_count = 0
    
    while cycle_count < 50:
        await RisingEdge(dut.clk)
        cycle_count += 1
        
        # Print detailed MAC trace for each cycle
        print(f"\n=== Cycle {cycle_count} ===")
        print(f"State: {int(dut.current_state.value) if hasattr(dut, 'current_state') else 'N/A'}")
        print(f"Input idx: {int(dut.input_idx.value) if hasattr(dut, 'input_idx') else 'N/A'}")
        print(f"Output idx: {int(dut.output_idx.value) if hasattr(dut, 'output_idx') else 'N/A'}")
        print(f"Weight idx: {int(dut.weight_idx.value) if hasattr(dut, 'weight_idx') else 'N/A'}")
        
        # MAC unit signals
        if hasattr(dut, 'mac_inst'):
            mac = dut.mac_inst
            print(f"MAC reset: {int(dut.mac_reset.value) if hasattr(dut, 'mac_reset') else 'N/A'}")
            print(f"MAC load_bias: {int(dut.mac_load_bias.value) if hasattr(dut, 'mac_load_bias') else 'N/A'}")
            print(f"MAC bias_in: {int(dut.mac_bias_in.value) if hasattr(dut, 'mac_bias_in') else 'N/A'}")
            print(f"MAC left_in: {int(dut.mac_left_in.value) if hasattr(dut, 'mac_left_in') else 'N/A'}")
            print(f"MAC top_in: {int(dut.mac_top_in.value) if hasattr(dut, 'mac_top_in') else 'N/A'}")
            print(f"MAC sum_out: {int(dut.mac_sum_out.value) if hasattr(dut, 'mac_sum_out') else 'N/A'}")
            
            # MAC internal signals if accessible
            if hasattr(mac, 'mult'):
                print(f"MAC mult: {int(mac.mult.value)}")
            if hasattr(mac, 'sum'):
                print(f"MAC sum: {int(mac.sum.value)}")
            if hasattr(mac, 'accumulator'):
                print(f"MAC accumulator: {int(mac.accumulator.value)}")
        
        # Memory interface signals
        print(f"Tensor RAM RE: {int(dut.tensor_ram_re.value)}")
        print(f"Weight ROM RE: {int(dut.weight_rom_re.value)}")
        print(f"Bias ROM RE: {int(dut.bias_rom_re.value)}")
        
        # Provide memory data based on addresses
        if int(dut.tensor_ram_re.value) == 1:
            addr = int(dut.tensor_ram_addr.value)
            if addr < len(input_vector):
                dut.tensor_ram_dout.value = input_vector[addr]
                print(f"  Tensor RAM read: addr={addr}, data={input_vector[addr]}")
            else:
                dut.tensor_ram_dout.value = 0
                print(f"  Tensor RAM read: addr={addr}, data=0 (out of bounds)")
                
        if int(dut.weight_rom_re.value) == 1:
            addr = int(dut.weight_rom_addr.value)
            if addr < len(weights_flat):
                dut.weight_rom_dout.value = weights_flat[addr]
                print(f"  Weight ROM read: addr={addr}, data={weights_flat[addr]}")
            else:
                dut.weight_rom_dout.value = 0
                print(f"  Weight ROM read: addr={addr}, data=0 (out of bounds)")
                
        if int(dut.bias_rom_re.value) == 1:
            addr = int(dut.bias_rom_addr.value)
            if addr < len(bias_vector):
                dut.bias_rom_dout.value = bias_vector[addr]
                print(f"  Bias ROM read: addr={addr}, data={bias_vector[addr]}")
            else:
                dut.bias_rom_dout.value = 0
                print(f"  Bias ROM read: addr={addr}, data=0 (out of bounds)")
        
        # Output signals
        print(f"Output ready: {int(dut.output_ready.value)}")
        print(f"Output data: {int(dut.output_data.value)}")
        print(f"Output channel: {int(dut.output_channel.value)}")
        print(f"Computation complete: {int(dut.computation_complete.value)}")
        
        # Check if output is ready
        if int(dut.output_ready.value) == 1:
            channel = int(dut.output_channel.value)
            output_data = int(dut.output_data.value)
            output_order.append((channel, output_data))
            print(f"Output {channel}: {output_data}")
        
        if int(dut.computation_complete.value) == 1:
            break
    
    # Verify outputs are produced in order (0, 1, 2)
    assert len(output_order) == 3, f"Expected 3 outputs, got {len(output_order)}"
    
    for i, (channel, data) in enumerate(output_order):
        assert channel == i, f"Output {i} should be channel {i}, got channel {channel}"
    
    print("✅ Multiple outputs timing test passed!")

if __name__ == "__main__":
    print("Dense layer compute testbench") 