import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Timer
from cocotb.regression import TestFactory
import random
import numpy as np

# Helper to load the image_data.hex file into a numpy array
HEX_FILE = '../rtl/image_data.hex'
def load_image_data():
    with open(HEX_FILE, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    # Each line is 32 hex chars = 16 bytes = 128 bits
    data = []
    for line in lines:
        for i in range(0, len(line), 2):
            data.append(int(line[i:i+2], 16))
    return np.array(data, dtype=np.uint8)

def is_padded_addr(ram_addr, img_w, img_h, pad, total_words):
    # For a 32x32 image with 1 pad, valid addresses are those that map to the image
    # Each ram_addr is a 128-bit word (16 bytes = 16 pixels for 1 channel)
    # The unified buffer may request addresses outside the image for padding
    # For this test, if ram_addr*16 >= img_w*img_h, it's padded
    base = ram_addr * 16
    if base >= img_w * img_h:
        return True
    return False

@cocotb.test()
async def test_a_lane_pattern(dut):
    """Test that verifies the correct A0-A3 data pattern for a single channel, with padding awareness"""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    dut._log.info("Testing A lane data pattern for single channel (with padding awareness)")
    
    # Configure for 32x32x1 image
    dut.img_width.value = 32
    dut.img_height.value = 32
    dut.num_channels.value = 1
    dut.pad_top.value = 1
    dut.pad_bottom.value = 1
    dut.pad_left.value = 1
    dut.pad_right.value = 1
    
    # Start extraction process
    dut.start_extraction.value = 1
    await RisingEdge(dut.clk)
    dut.start_extraction.value = 0
    
    # Wait for first block to be ready
    while not dut.block_ready.value:
        await RisingEdge(dut.clk)
        # Check that tensor_ram_inst.we is always zero
        assert int(dut.tensor_ram_inst.we.value) == 0, "tensor_ram_inst.we should always be zero during test"
        dut._log.info(f"Waiting for block_ready... ram_re: {dut.unified_buffer_inst.ram_re.value}, ram_addr: {dut.unified_buffer_inst.ram_addr.value}")
    
    dut._log.info("First block loaded and ready")
    
    # Start spatial data formatting
    dut.start_formatting.value = 1
    await RisingEdge(dut.clk)
    dut.start_formatting.value = 0
    
    # Wait for first valid formatted data
    while not dut.formatted_data_valid.value:
        await RisingEdge(dut.clk)
        # Check that tensor_ram_inst.we is always zero
        assert int(dut.tensor_ram_inst.we.value) == 0, "tensor_ram_inst.we should always be zero during test"
        dut._log.info(f"Waiting for formatted_data_valid...")
    
    dut._log.info("Formatted data valid, starting pattern verification")
    
    # Load the image data for expected value checking
    image_data = load_image_data()
    IMG_W = 32
    IMG_H = 32
    PAD = 1
    TOTAL_WORDS = (IMG_W * IMG_H + 15) // 16
    
    # Monitor formatted outputs for several cycles
    formatted_outputs = []
    for cycle in range(35):  # Run for enough cycles to see the complete pattern
        await RisingEdge(dut.clk)
        # Check that tensor_ram_inst.we is always zero
        assert int(dut.tensor_ram_inst.we.value) == 0, f"tensor_ram_inst.we should always be zero during test (cycle {cycle})"
        
        # Log buffer state
        ram_re = int(dut.unified_buffer_inst.ram_re.value)
        ram_addr = int(dut.unified_buffer_inst.ram_addr.value)
        dut._log.info(f"Cycle {cycle} buffer state:")
        dut._log.info(f"  ram_re: {ram_re}")
        dut._log.info(f"  ram_addr: {ram_addr:08x}")
        dut._log.info(f"  ram_dout0: {dut.unified_buffer_inst.ram_dout0.value}")
        dut._log.info(f"  ram_dout1: {dut.unified_buffer_inst.ram_dout1.value}")
        dut._log.info(f"  ram_dout2: {dut.unified_buffer_inst.ram_dout2.value}")
        dut._log.info(f"  ram_dout3: {dut.unified_buffer_inst.ram_dout3.value}")
        # Print expected value from memory if ram_re is high
        if ram_re:
            if is_padded_addr(ram_addr, IMG_W, IMG_H, PAD, TOTAL_WORDS):
                dut._log.info(f"  [EXPECTED] PADDING: All bytes should be 00")
            else:
                base = ram_addr * 16
                expected_bytes = image_data[base:base+16]
                expected_hex = ''.join(f'{b:02x}' for b in expected_bytes)
                dut._log.info(f"  [EXPECTED] image_data.hex[{base}:{base+16}]: {expected_hex}")
        
        if dut.formatted_data_valid.value:
            # Capture formatted data
            a0_data = [int(dut.formatted_A0_0.value), int(dut.formatted_A0_1.value), 
                      int(dut.formatted_A0_2.value), int(dut.formatted_A0_3.value)]
            a1_data = [int(dut.formatted_A1_0.value), int(dut.formatted_A1_1.value),
                      int(dut.formatted_A1_2.value), int(dut.formatted_A1_3.value)]
            a2_data = [int(dut.formatted_A2_0.value), int(dut.formatted_A2_1.value),
                      int(dut.formatted_A2_2.value), int(dut.formatted_A2_3.value)]
            a3_data = [int(dut.formatted_A3_0.value), int(dut.formatted_A3_1.value),
                      int(dut.formatted_A3_2.value), int(dut.formatted_A3_3.value)]
            
            formatted_outputs.append({
                'cycle': cycle,
                'A0': a0_data,
                'A1': a1_data, 
                'A2': a2_data,
                'A3': a3_data
            })
            
            # Verify single channel constraint
            active_channels = [a0_data[0], a1_data[0], a2_data[0], a3_data[0]]
            inactive_channels = (a0_data[1:] + a1_data[1:] + a2_data[1:] + a3_data[1:])
            
            # Check if inactive channels are all zero
            inactive_all_zero = all(val == 0 for val in inactive_channels)
            if not inactive_all_zero:
                dut._log.error(f"Cycle {cycle}: Inactive channels not zero!")
                dut._log.error(f"  Active: {[f'{v:02x}' for v in active_channels]}")
                dut._log.error(f"  Inactive: {[f'{v:02x}' for v in inactive_channels]}")
                assert False, "Inactive channels must be zero for single channel"
            
            # Calculate expected row and column for each A lane
            cycle_in_row = cycle % 7
            row_base = cycle // 7
            
            # Expected row for each A lane
            a_rows = [row_base, row_base + 1, row_base + 2, row_base + 3]
            a_datas = [a0_data[0], a1_data[0], a2_data[0], a3_data[0]]
            lane_names = ['A0', 'A1', 'A2', 'A3']
            
            # Log the pattern for debugging
            dut._log.info(f"Cycle {cycle} formatted data:")
            for lane_idx, (row, data, name) in enumerate(zip(a_rows, a_datas, lane_names)):
                dut._log.info(f"  {name}[0] at row {row}, col {cycle_in_row}: {data:02x}")
                
                # Determine if this lane should be active in this cycle
                is_lane_active = (row >= 0 and row < 7 and cycle >= lane_idx)
                
                if is_lane_active:
                    # Lane should be active and match its row value
                    if data != row:
                        dut._log.error(f"Cycle {cycle}: {name}[0] value {data:02x} does not match expected row value {row:02x}")
                        assert False, f"{name} data pattern incorrect"
                else:
                    # Lane should be inactive (zero)
                    if data != 0:
                        dut._log.error(f"Cycle {cycle}: {name}[0] should be zero, got {data:02x}")
                        assert False, f"{name} should be zero when inactive"
    
    # Verify we got some formatted output
    assert len(formatted_outputs) > 0, "No formatted data was output"
    dut._log.info(f"Captured {len(formatted_outputs)} cycles of formatted data")
    
    # Verify that non-zero data appears
    has_nonzero_data = False
    for output in formatted_outputs:
        if any(val != 0 for row in [output['A0'], output['A1'], output['A2'], output['A3']] for val in row):
            has_nonzero_data = True
            break
    
    assert has_nonzero_data, "All formatted output was zero - data may not be flowing correctly"
    
    dut._log.info("A lane pattern test completed successfully!")
    dut._log.info("✓ Single channel constraint verified")
    dut._log.info("✓ A0-A3 row sequence verified")
    dut._log.info("✓ Data pattern matches expected values")

# Factory to run tests
factory = TestFactory(test_a_lane_pattern)
factory.generate_tests() 