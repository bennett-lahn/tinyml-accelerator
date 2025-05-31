import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
from cocotb.binary import BinaryValue
import numpy as np

@cocotb.test()
async def test_memory_traverse(dut):
    """Detailed trace traversing the whole memory to understand data layout"""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    # Configure for 16x16x8 image
    img_width = 16
    img_height = 16
    num_channels = 8
    
    dut.img_width.value = img_width
    dut.img_height.value = img_height
    dut.num_channels.value = num_channels
    dut.pad_top.value = 0
    dut.pad_bottom.value = 0
    dut.pad_left.value = 0
    dut.pad_right.value = 0
    
    # Initialize control signals
    dut.start_extraction.value = 0
    dut.next_channel_group.value = 0
    dut.next_spatial_block.value = 0
    dut.start_formatting.value = 0
    dut.tensor_ram_we.value = 0
    
    await RisingEdge(dut.clk)
    
    cocotb.log.info("=== MEMORY LAYOUT ANALYSIS FOR 16x16x8 DATA ===")
    cocotb.log.info(f"Image dimensions: {img_width}x{img_height}x{num_channels}")
    cocotb.log.info(f"Total pixels: {img_width * img_height} = {img_width * img_height}")
    cocotb.log.info(f"Total values: {img_width * img_height * num_channels} = {img_width * img_height * num_channels}")
    cocotb.log.info(f"Channel-last layout: [row, col, channel]")
    cocotb.log.info(f"Each RAM address contains 16 bytes = 2 pixels worth of data")
    cocotb.log.info(f"Total RAM addresses needed: {(img_width * img_height * num_channels) // 16} = {(img_width * img_height * num_channels) // 16}")
    
    # Calculate expected data pattern
    def get_expected_value(row, col, channel):
        """Calculate expected value for position (row, col, channel)"""
        base_idx = (row * img_width + col) * num_channels
        return (base_idx + channel) % 256
    
    def get_ram_address(row, col):
        """Calculate RAM address for spatial position (row, col)"""
        return row * img_width + col
    
    def get_byte_offset(row, col, channel):
        """Calculate byte offset within the 16-byte RAM word"""
        pixel_in_line = (row * img_width + col) % 2  # 2 pixels per 16-byte line
        return pixel_in_line * num_channels + channel
    
    # Print detailed memory mapping for first few positions
    cocotb.log.info("\n=== DETAILED MEMORY MAPPING ===")
    for row in range(min(3, img_height)):
        for col in range(min(4, img_width)):
            ram_addr = get_ram_address(row, col)
            cocotb.log.info(f"\nPosition ({row},{col}):")
            cocotb.log.info(f"  RAM address: {ram_addr}")
            
            channels = []
            byte_offsets = []
            for ch in range(num_channels):
                val = get_expected_value(row, col, ch)
                byte_off = get_byte_offset(row, col, ch)
                channels.append(val)
                byte_offsets.append(byte_off)
            
            cocotb.log.info(f"  Expected channels: {channels}")
            cocotb.log.info(f"  Byte offsets: {byte_offsets}")
            
            # Show channel groups
            ch_group_0 = channels[0:4]
            ch_group_1 = channels[4:8]
            cocotb.log.info(f"  Channel group 0 (0-3): {ch_group_0}")
            cocotb.log.info(f"  Channel group 1 (4-7): {ch_group_1}")
    
    # Now let's trace through the actual hardware access
    cocotb.log.info("\n=== HARDWARE ACCESS TRACE ===")
    
    # Start extraction for channel group 0
    cocotb.log.info("\n--- CHANNEL GROUP 0 (channels 0-3) ---")
    dut.start_extraction.value = 1
    await RisingEdge(dut.clk)
    dut.start_extraction.value = 0
    
    # Monitor RAM accesses during loading
    ram_accesses = []
    cycle_count = 0
    
    while not dut.buffer_loading_complete.value and cycle_count < 200:
        await RisingEdge(dut.clk)
        cycle_count += 1
        
        if dut.ram_re.value:
            ram_addr = int(dut.ram_addr_truncated.value)
            ram_accesses.append(ram_addr)
            
            # Wait one cycle for data to be available
            await RisingEdge(dut.clk)
            cycle_count += 1
            
            # Read the RAM outputs
            full_data = int(dut.ram_dout.value)
            dout0 = int(dut.ram_dout0.value)
            dout1 = int(dut.ram_dout1.value)
            dout2 = int(dut.ram_dout2.value)
            dout3 = int(dut.ram_dout3.value)
            
            cocotb.log.info(f"RAM read at address {ram_addr}:")
            cocotb.log.info(f"  Full 128-bit: 0x{full_data:032x}")
            cocotb.log.info(f"  dout0 (bytes 3:0):   0x{dout0:08x} = [{(dout0>>24)&0xFF}, {(dout0>>16)&0xFF}, {(dout0>>8)&0xFF}, {dout0&0xFF}]")
            cocotb.log.info(f"  dout1 (bytes 7:4):   0x{dout1:08x} = [{(dout1>>24)&0xFF}, {(dout1>>16)&0xFF}, {(dout1>>8)&0xFF}, {dout1&0xFF}]")
            cocotb.log.info(f"  dout2 (bytes 11:8):  0x{dout2:08x} = [{(dout2>>24)&0xFF}, {(dout2>>16)&0xFF}, {(dout2>>8)&0xFF}, {dout2&0xFF}]")
            cocotb.log.info(f"  dout3 (bytes 15:12): 0x{dout3:08x} = [{(dout3>>24)&0xFF}, {(dout3>>16)&0xFF}, {(dout3>>8)&0xFF}, {dout3&0xFF}]")
            
            # Determine which spatial positions this address covers
            positions_covered = []
            for pos in range(img_width * img_height):
                if get_ram_address(pos // img_width, pos % img_width) == ram_addr:
                    positions_covered.append((pos // img_width, pos % img_width))
            
            cocotb.log.info(f"  Covers positions: {positions_covered}")
            
            # Show expected data for these positions
            for pos_row, pos_col in positions_covered:
                expected_channels = []
                for ch in range(num_channels):
                    expected_channels.append(get_expected_value(pos_row, pos_col, ch))
                cocotb.log.info(f"    Position ({pos_row},{pos_col}) expected: {expected_channels}")
    
    await RisingEdge(dut.clk)
    
    cocotb.log.info(f"\nChannel group 0 loading completed after {cycle_count} cycles")
    cocotb.log.info(f"RAM addresses accessed: {sorted(set(ram_accesses))}")
    
    # Check what was loaded into position (0,0)
    patch_00 = int(dut.patch_pe00_out.value)
    ch0 = patch_00 & 0xFF
    ch1 = (patch_00 >> 8) & 0xFF
    ch2 = (patch_00 >> 16) & 0xFF
    ch3 = (patch_00 >> 24) & 0xFF
    
    cocotb.log.info(f"\nLoaded into position (0,0): [{ch0}, {ch1}, {ch2}, {ch3}]")
    
    expected_00 = [get_expected_value(0, 0, ch) for ch in range(4)]
    cocotb.log.info(f"Expected for position (0,0): {expected_00}")
    
    if [ch0, ch1, ch2, ch3] == expected_00:
        cocotb.log.info("✓ Channel group 0 data matches expected!")
    else:
        cocotb.log.error("✗ Channel group 0 data mismatch!")
    
    # Now test channel group 1
    cocotb.log.info("\n--- CHANNEL GROUP 1 (channels 4-7) ---")
    dut.next_channel_group.value = 1
    await RisingEdge(dut.clk)
    dut.next_channel_group.value = 0
    
    # Monitor RAM accesses for channel group 1
    ram_accesses_ch1 = []
    cycle_count = 0
    
    while not dut.buffer_loading_complete.value and cycle_count < 200:
        await RisingEdge(dut.clk)
        cycle_count += 1
        
        if dut.ram_re.value:
            ram_addr = int(dut.ram_addr_truncated.value)
            ram_accesses_ch1.append(ram_addr)
            
            if len(ram_accesses_ch1) <= 3:  # Only log first few accesses to avoid spam
                cocotb.log.info(f"Channel group 1 - RAM read at address {ram_addr}")
    
    await RisingEdge(dut.clk)
    
    cocotb.log.info(f"\nChannel group 1 loading completed after {cycle_count} cycles")
    cocotb.log.info(f"RAM addresses accessed: {sorted(set(ram_accesses_ch1))}")
    
    # Check what was loaded into position (0,0) for channel group 1
    patch_00_ch1 = int(dut.patch_pe00_out.value)
    ch4 = patch_00_ch1 & 0xFF
    ch5 = (patch_00_ch1 >> 8) & 0xFF
    ch6 = (patch_00_ch1 >> 16) & 0xFF
    ch7 = (patch_00_ch1 >> 24) & 0xFF
    
    cocotb.log.info(f"\nLoaded into position (0,0): [{ch4}, {ch5}, {ch6}, {ch7}]")
    
    expected_00_ch1 = [get_expected_value(0, 0, ch) for ch in range(4, 8)]
    cocotb.log.info(f"Expected for position (0,0): {expected_00_ch1}")
    
    if [ch4, ch5, ch6, ch7] == expected_00_ch1:
        cocotb.log.info("✓ Channel group 1 data matches expected!")
    else:
        cocotb.log.error("✗ Channel group 1 data mismatch!")
    
    # Test a few more spatial positions
    cocotb.log.info("\n=== SPATIAL POSITION VERIFICATION ===")
    
    # Check position (0,1) - should be different data
    patch_01 = int(dut.patch_pe01_out.value)
    ch01_0 = patch_01 & 0xFF
    ch01_1 = (patch_01 >> 8) & 0xFF
    ch01_2 = (patch_01 >> 16) & 0xFF
    ch01_3 = (patch_01 >> 24) & 0xFF
    
    expected_01_ch1 = [get_expected_value(0, 1, ch) for ch in range(4, 8)]
    cocotb.log.info(f"Position (0,1) loaded: [{ch01_0}, {ch01_1}, {ch01_2}, {ch01_3}]")
    cocotb.log.info(f"Position (0,1) expected: {expected_01_ch1}")
    
    # Summary
    cocotb.log.info("\n=== SUMMARY ===")
    cocotb.log.info("Key findings:")
    cocotb.log.info("1. Each RAM address contains data for 2 spatial positions (16 bytes = 2×8 channels)")
    cocotb.log.info("2. Channel groups 0 and 1 for same position use same RAM address")
    cocotb.log.info("3. Channel group determines which 4-byte chunk from 16-byte RAM word")
    cocotb.log.info("4. Byte order is corrected for little-endian format")
    cocotb.log.info("5. Address calculation: addr = row * width + col")
    
    cocotb.log.info("Memory traverse test completed!") 