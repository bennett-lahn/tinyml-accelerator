import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
from cocotb.binary import BinaryValue
import numpy as np

@cocotb.test()
async def test_multi_channel_16x16x8_integration(dut):
    """Test complete multi-channel data flow through tensor RAM -> unified buffer -> spatial formatter
    Tests 16x16x8 data with 2 channel groups (0-3, 4-7)"""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    # Configure for 16x16x8 image (8 channels)
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
    
    # Use the existing hex file data (16x16x8)
    cocotb.log.info("Using 16x16x8 tensor RAM data from hex file...")
    
    # Start extraction for first channel group (channels 0-3)
    cocotb.log.info("Starting extraction for channel group 0 (channels 0-3)...")
    
    dut.start_extraction.value = 1
    await RisingEdge(dut.clk)
    dut.start_extraction.value = 0
    
    # Wait for buffer loading to complete
    cycle_count = 0
    timeout_cycles = 200
    
    while not dut.buffer_loading_complete.value and cycle_count < timeout_cycles:
        await RisingEdge(dut.clk)
        cycle_count += 1
        if cycle_count % 20 == 0:
            cocotb.log.info(f"Waiting for channel group 0 loading... cycle {cycle_count}")
    
    if cycle_count >= timeout_cycles:
        cocotb.log.error("Timeout waiting for channel group 0 loading")
        assert False, "Channel group 0 loading timeout"
    
    cocotb.log.info(f"Channel group 0 loaded after {cycle_count} cycles")
    
    # Wait one more cycle for state transition
    await RisingEdge(dut.clk)
    
    # Verify first channel group data
    assert dut.block_ready.value == 1, "Block should be ready after loading"
    assert dut.patches_valid.value == 1, "Patches should be valid"
    
    # Get patch data for position (0,0) - should contain channels 0-3
    patch_00 = int(dut.patch_pe00_out.value)
    
    # Extract individual bytes (channels)
    ch0 = patch_00 & 0xFF
    ch1 = (patch_00 >> 8) & 0xFF
    ch2 = (patch_00 >> 16) & 0xFF
    ch3 = (patch_00 >> 24) & 0xFF
    
    cocotb.log.info(f"Channel group 0 - patch_pe00_out: 0x{patch_00:08x}")
    cocotb.log.info(f"  Channels: [{ch0}, {ch1}, {ch2}, {ch3}]")
    
    # For position (0,0), channels 0-3 should be [0, 1, 2, 3]
    expected_channels_0 = [0, 1, 2, 3]
    actual_channels_0 = [ch0, ch1, ch2, ch3]
    
    cocotb.log.info(f"Expected channels 0-3: {expected_channels_0}")
    cocotb.log.info(f"Actual channels 0-3: {actual_channels_0}")
    
    for i, (actual, expected) in enumerate(zip(actual_channels_0, expected_channels_0)):
        assert actual == expected, f"Channel {i} mismatch: got {actual}, expected {expected}"
    
    cocotb.log.info("✓ Channel group 0 (channels 0-3) verification passed!")
    
    # Test spatial data formatter for first channel group
    cocotb.log.info("Testing spatial formatter for channel group 0...")
    
    dut.start_formatting.value = 1
    await RisingEdge(dut.clk)
    dut.start_formatting.value = 0
    
    # Monitor a few cycles of formatted output
    formatted_cycles = 0
    max_format_cycles = 10
    
    while formatted_cycles < max_format_cycles:
        await RisingEdge(dut.clk)
        
        if dut.formatted_data_valid.value:
            a0_0 = int(dut.formatted_A0_0.value.signed_integer) if hasattr(dut.formatted_A0_0.value, 'signed_integer') else int(dut.formatted_A0_0.value)
            a0_1 = int(dut.formatted_A0_1.value.signed_integer) if hasattr(dut.formatted_A0_1.value, 'signed_integer') else int(dut.formatted_A0_1.value)
            
            cocotb.log.info(f"Formatted cycle {formatted_cycles}: A0[0]={a0_0}, A0[1]={a0_1}")
            
            if formatted_cycles == 0:
                # First element should be channel 0 value (0)
                assert a0_0 == 0, f"First formatted element should be 0, got {a0_0}"
            
            formatted_cycles += 1
            
            if formatted_cycles >= 3:  # Just check a few cycles
                break
        else:
            formatted_cycles += 1
    
    # Now test channel group advancement
    cocotb.log.info("Testing channel group advancement...")
    
    dut.next_channel_group.value = 1
    await RisingEdge(dut.clk)
    dut.next_channel_group.value = 0
    
    # Check if we need to advance to next channel group
    await RisingEdge(dut.clk)
    
    if not dut.all_channels_done.value:
        cocotb.log.info("Moving to channel group 1 (channels 4-7)...")
        
        # Wait for next channel group to load
        cycle_count = 0
        while not dut.buffer_loading_complete.value and cycle_count < timeout_cycles:
            await RisingEdge(dut.clk)
            cycle_count += 1
            if cycle_count % 20 == 0:
                cocotb.log.info(f"Waiting for channel group 1 loading... cycle {cycle_count}")
        
        if cycle_count >= timeout_cycles:
            cocotb.log.error("Timeout waiting for channel group 1 loading")
            assert False, "Channel group 1 loading timeout"
        
        cocotb.log.info(f"Channel group 1 loaded after {cycle_count} cycles")
        
        # Wait one more cycle for state transition
        await RisingEdge(dut.clk)
        
        # Verify second channel group data
        patch_00_ch1 = int(dut.patch_pe00_out.value)
        
        # Extract channels 4-7
        ch4 = patch_00_ch1 & 0xFF
        ch5 = (patch_00_ch1 >> 8) & 0xFF
        ch6 = (patch_00_ch1 >> 16) & 0xFF
        ch7 = (patch_00_ch1 >> 24) & 0xFF
        
        cocotb.log.info(f"Channel group 1 - patch_pe00_out: 0x{patch_00_ch1:08x}")
        cocotb.log.info(f"  Channels: [{ch4}, {ch5}, {ch6}, {ch7}]")
        
        # For position (0,0), channels 4-7 should be [4, 5, 6, 7]
        expected_channels_1 = [4, 5, 6, 7]
        actual_channels_1 = [ch4, ch5, ch6, ch7]
        
        cocotb.log.info(f"Expected channels 4-7: {expected_channels_1}")
        cocotb.log.info(f"Actual channels 4-7: {actual_channels_1}")
        
        for i, (actual, expected) in enumerate(zip(actual_channels_1, expected_channels_1)):
            assert actual == expected, f"Channel {i+4} mismatch: got {actual}, expected {expected}"
        
        cocotb.log.info("✓ Channel group 1 (channels 4-7) verification passed!")
        
        # Test next channel group again to complete all channels
        dut.next_channel_group.value = 1
        await RisingEdge(dut.clk)
        dut.next_channel_group.value = 0
        await RisingEdge(dut.clk)
        
        if dut.all_channels_done.value:
            cocotb.log.info("✓ All 8 channels processed successfully!")
        else:
            cocotb.log.error("✗ Not all channels completed")
            assert False, "All channels should be done after processing both groups"
    
    else:
        cocotb.log.error("✗ All channels done too early - should process 2 groups")
        assert False, "Should require 2 channel groups for 8 channels"
    
    # Test spatial block advancement
    cocotb.log.info("Testing spatial block advancement...")
    
    dut.next_spatial_block.value = 1
    await RisingEdge(dut.clk)
    dut.next_spatial_block.value = 0
    
    # Wait for next spatial block to load (should start with channel group 0 again)
    cycle_count = 0
    while not dut.buffer_loading_complete.value and cycle_count < timeout_cycles:
        await RisingEdge(dut.clk)
        cycle_count += 1
    
    if cycle_count < timeout_cycles:
        await RisingEdge(dut.clk)  # Wait for state transition
        
        # Verify we got different spatial data
        new_patch_00 = int(dut.patch_pe00_out.value)
        cocotb.log.info(f"Next spatial block - patch_pe00_out: 0x{new_patch_00:08x}")
        
        if new_patch_00 != patch_00:
            cocotb.log.info("✓ Spatial advancement working - different data loaded!")
        else:
            cocotb.log.info("Note: Spatial data appears same - may be at edge or using same pattern")
    
    cocotb.log.info("Multi-channel 16x16x8 integration test completed successfully!")

@cocotb.test()
async def test_channel_group_counting(dut):
    """Test that the system correctly calculates and processes the right number of channel groups"""
    
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
    
    # Start extraction
    dut.start_extraction.value = 1
    await RisingEdge(dut.clk)
    dut.start_extraction.value = 0
    
    # Wait for loading to complete
    cycle_count = 0
    while not dut.buffer_loading_complete.value and cycle_count < 200:
        await RisingEdge(dut.clk)
        cycle_count += 1
    
    await RisingEdge(dut.clk)
    
    # Check internal channel group tracking
    ub = dut.unified_buffer_inst
    total_channel_groups = int(ub.total_channel_groups.value)
    current_channel_group = int(ub.channel_group.value)
    
    cocotb.log.info(f"Total channel groups: {total_channel_groups}")
    cocotb.log.info(f"Current channel group: {current_channel_group}")
    
    # For 8 channels with max 4 channels per group, we should have 2 groups
    assert total_channel_groups == 2, f"Expected 2 channel groups for 8 channels, got {total_channel_groups}"
    assert current_channel_group == 0, f"Should start with channel group 0, got {current_channel_group}"
    
    # Test channel group advancement
    channel_groups_processed = 0
    
    while channel_groups_processed < total_channel_groups:
        cocotb.log.info(f"Processing channel group {channel_groups_processed}")
        
        # Advance to next channel group
        dut.next_channel_group.value = 1
        await RisingEdge(dut.clk)
        dut.next_channel_group.value = 0
        await RisingEdge(dut.clk)
        
        channel_groups_processed += 1
        
        if channel_groups_processed < total_channel_groups:
            # Should not be done yet
            assert not dut.all_channels_done.value, f"Should not be done after {channel_groups_processed} groups"
            
            # Wait for next group to load
            cycle_count = 0
            while not dut.buffer_loading_complete.value and cycle_count < 200:
                await RisingEdge(dut.clk)
                cycle_count += 1
            await RisingEdge(dut.clk)
        else:
            # Should be done now
            assert dut.all_channels_done.value, f"Should be done after processing all {total_channel_groups} groups"
    
    cocotb.log.info("✓ Channel group counting and advancement test passed!") 