import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
from cocotb.binary import BinaryValue
import numpy as np

@cocotb.test()
async def test_endianness_debug(dut):
    """Debug test to understand the exact data layout and endianness"""
    
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
    dut.img_width.value = 16
    dut.img_height.value = 16
    dut.num_channels.value = 8
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
    
    # Let's examine what's in the tensor RAM at addresses 0 and 1
    cocotb.log.info("=== TENSOR RAM DEBUG ===")
    
    # Access tensor RAM through the integration module's internal signals
    tensor_ram = dut.tensor_ram_inst
    
    # Force a read from tensor RAM at address 0 by setting the internal signals
    # We'll trigger this by starting extraction which will cause RAM reads
    dut.start_extraction.value = 1
    await RisingEdge(dut.clk)
    dut.start_extraction.value = 0
    
    # Wait a few cycles for the first RAM read to happen
    for i in range(10):
        await RisingEdge(dut.clk)
        
        # Check if RAM is being read
        if dut.ram_re.value:
            ram_addr = int(dut.ram_addr_truncated.value)
            cocotb.log.info(f"RAM read at address {ram_addr}")
            
            # Wait one more cycle for data to be available
            await RisingEdge(dut.clk)
            
            # Read the outputs
            full_data = int(dut.ram_dout.value)
            dout0 = int(dut.ram_dout0.value)
            dout1 = int(dut.ram_dout1.value)
            dout2 = int(dut.ram_dout2.value)
            dout3 = int(dut.ram_dout3.value)
            
            cocotb.log.info(f"RAM address {ram_addr}:")
            cocotb.log.info(f"  Full 128-bit data: 0x{full_data:032x}")
            cocotb.log.info(f"  dout0 (bytes 3:0):  0x{dout0:08x}")
            cocotb.log.info(f"  dout1 (bytes 7:4):  0x{dout1:08x}")
            cocotb.log.info(f"  dout2 (bytes 11:8): 0x{dout2:08x}")
            cocotb.log.info(f"  dout3 (bytes 15:12): 0x{dout3:08x}")
            
            # Extract individual bytes from each dout
            for j, dout_val in enumerate([dout0, dout1, dout2, dout3]):
                byte0 = dout_val & 0xFF
                byte1 = (dout_val >> 8) & 0xFF
                byte2 = (dout_val >> 16) & 0xFF
                byte3 = (dout_val >> 24) & 0xFF
                base_byte_idx = j * 4
                cocotb.log.info(f"  dout{j} bytes: [{base_byte_idx}]={byte0}, [{base_byte_idx+1}]={byte1}, [{base_byte_idx+2}]={byte2}, [{base_byte_idx+3}]={byte3}")
            
            if ram_addr >= 1:  # Stop after we've seen address 1
                break
    
    # Wait for buffer loading to complete
    cycle_count = 0
    while not dut.buffer_loading_complete.value and cycle_count < 200:
        await RisingEdge(dut.clk)
        cycle_count += 1
    
    await RisingEdge(dut.clk)
    
    # Now test the unified buffer extraction
    cocotb.log.info("\n=== UNIFIED BUFFER DEBUG - CHANNEL GROUP 0 ===")
    
    # Check what the unified buffer loaded for position (0,0)
    patch_00 = int(dut.patch_pe00_out.value)
    cocotb.log.info(f"Unified buffer patch_pe00_out: 0x{patch_00:08x}")
    
    # Extract individual bytes (channels)
    ch0 = patch_00 & 0xFF
    ch1 = (patch_00 >> 8) & 0xFF
    ch2 = (patch_00 >> 16) & 0xFF
    ch3 = (patch_00 >> 24) & 0xFF
    
    cocotb.log.info(f"Extracted channels: [{ch0}, {ch1}, {ch2}, {ch3}]")
    
    # Let's also check the unified buffer's internal signals
    ub = dut.unified_buffer_inst
    current_channel_group = int(ub.channel_group.value)
    total_channel_groups = int(ub.total_channel_groups.value)
    
    cocotb.log.info(f"Channel group: {current_channel_group}/{total_channel_groups}")
    
    # Check the address calculation
    if hasattr(ub, 'calculated_addr'):
        calc_addr = int(ub.calculated_addr.value)
        cocotb.log.info(f"Calculated address for position (0,0): {calc_addr}")
    
    # Now test channel group advancement
    cocotb.log.info("\n=== CHANNEL GROUP ADVANCEMENT DEBUG ===")
    
    dut.next_channel_group.value = 1
    await RisingEdge(dut.clk)
    dut.next_channel_group.value = 0
    await RisingEdge(dut.clk)
    
    # Check the new channel group
    new_channel_group = int(ub.channel_group.value)
    cocotb.log.info(f"After advancement - Channel group: {new_channel_group}/{total_channel_groups}")
    
    # Wait for buffer loading to complete for channel group 1
    cycle_count = 0
    while not dut.buffer_loading_complete.value and cycle_count < 200:
        await RisingEdge(dut.clk)
        cycle_count += 1
        if cycle_count % 20 == 0:
            cocotb.log.info(f"Waiting for channel group 1 loading... cycle {cycle_count}")
    
    await RisingEdge(dut.clk)
    
    cocotb.log.info("\n=== UNIFIED BUFFER DEBUG - CHANNEL GROUP 1 ===")
    
    # Check what the unified buffer loaded for position (0,0) in channel group 1
    patch_00_ch1 = int(dut.patch_pe00_out.value)
    cocotb.log.info(f"Channel group 1 - patch_pe00_out: 0x{patch_00_ch1:08x}")
    
    # Extract individual bytes (channels)
    ch4 = patch_00_ch1 & 0xFF
    ch5 = (patch_00_ch1 >> 8) & 0xFF
    ch6 = (patch_00_ch1 >> 16) & 0xFF
    ch7 = (patch_00_ch1 >> 24) & 0xFF
    
    cocotb.log.info(f"Extracted channels: [{ch4}, {ch5}, {ch6}, {ch7}]")
    
    # Check the address calculation for channel group 1
    if hasattr(ub, 'calculated_addr'):
        calc_addr_ch1 = int(ub.calculated_addr.value)
        cocotb.log.info(f"Calculated address for position (0,0), channel group 1: {calc_addr_ch1}")
    
    # Let's manually calculate what the address should be for channel group 1
    # For position (0,0), channel group 1:
    # addr = (row * width + col) * total_channel_groups + channel_group
    # addr = (0 * 16 + 0) * 2 + 1 = 1
    expected_addr_ch1 = 1
    cocotb.log.info(f"Expected address for position (0,0), channel group 1: {expected_addr_ch1}")
    
    # Let's analyze the data layout
    cocotb.log.info("\n=== DATA LAYOUT ANALYSIS ===")
    cocotb.log.info("According to generate_hex_file.py:")
    cocotb.log.info("- 16x16x8 = 2048 total pixels")
    cocotb.log.info("- Channel-last layout: [row, col, channel]")
    cocotb.log.info("- Position (0,0) should have channels [0,1,2,3,4,5,6,7]")
    cocotb.log.info("- Position (0,1) should have channels [8,9,10,11,12,13,14,15]")
    cocotb.log.info("- 16 pixels per 128-bit line (16 bytes per line)")
    cocotb.log.info("- So first line (address 0) contains pixels (0,0) and (0,1)")
    cocotb.log.info("- Address 0: bytes [0-15] = channels for (0,0) and (0,1)")
    cocotb.log.info("- Address 1: bytes [16-31] = channels for (0,2) and (0,3)")
    
    cocotb.log.info("\nExpected data mapping:")
    cocotb.log.info("Address 0 (bytes 0-15):")
    cocotb.log.info("  Position (0,0): channels [0,1,2,3,4,5,6,7] = bytes [0-7]")
    cocotb.log.info("  Position (0,1): channels [8,9,10,11,12,13,14,15] = bytes [8-15]")
    cocotb.log.info("Address 1 (bytes 16-31):")
    cocotb.log.info("  Position (0,2): channels [16,17,18,19,20,21,22,23] = bytes [16-23]")
    cocotb.log.info("  Position (0,3): channels [24,25,26,27,28,29,30,31] = bytes [24-31]")
    
    cocotb.log.info("\nFor position (0,0):")
    cocotb.log.info("  Channel group 0 (channels 0-3): should be bytes [0,1,2,3] from address 0")
    cocotb.log.info("  Channel group 1 (channels 4-7): should be bytes [4,5,6,7] from address 0")
    
    cocotb.log.info("\nBut our address calculation gives:")
    cocotb.log.info("  Channel group 0: address = (0*16+0)*2+0 = 0 ✓")
    cocotb.log.info("  Channel group 1: address = (0*16+0)*2+1 = 1 ✗ (should also be 0)")
    
    cocotb.log.info("\nThe issue is in the address calculation!")
    cocotb.log.info("For multi-channel data, all channel groups for the same position should use the same address")
    cocotb.log.info("The channel group should only affect which 32-bit chunk we select from the 128-bit data")
    
    cocotb.log.info("Endianness debug test completed!") 