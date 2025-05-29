import cocotb
import logging
from cocotb.triggers import RisingEdge
from cocotb.clock import Clock

# Set up logging
cocotb.log.setLevel(logging.INFO)

@cocotb.test()
async def test_detailed_channel_trace(dut):
    """Detailed trace showing every channel at every position for 16x16x8 tensor"""
    CLOCK_PERIOD = 2
    dut._log.setLevel(logging.INFO)
    
    # Start clock
    cocotb.start_soon(Clock(dut.clk, CLOCK_PERIOD, units="ns").start())
    
    async def tick():
        """Advance one clock cycle"""
        await RisingEdge(dut.clk)
    
    async def reset_dut():
        """Reset the DUT"""
        dut.reset.value = 1
        await tick()
        dut.reset.value = 0
        await tick()
    
    # Reset and configure for 16x16x8
    await reset_dut()
    
    dut.current_img_width.value = 16
    dut.current_img_height.value = 16  
    dut.current_num_channels.value = 8
    
    dut.start_patch_generation.value = 0
    dut.start_sliding_window.value = 0
    dut.advance_to_next_patch.value = 0
    
    await tick()
    
    dut._log.info("=== DETAILED CHANNEL TRACE: 16x16x8 Tensor ===")
    dut._log.info("Expected: 13x13=169 positions, 8 channels each = 1352 total extractions")
    dut._log.info("Format: [Patch#] Position(row,col) Channel=X [Complete=Y/N]")
    dut._log.info("=" * 80)
    
    # Track positions and channels
    position_channel_map = {}  # {(row,col): [list of channels seen]}
    patch_count = 0
    current_position = None
    
    # Start with first patch/channel
    dut.start_patch_generation.value = 1
    await tick()
    dut.start_patch_generation.value = 0
    
    # Wait for first patch
    timeout = 0
    while dut.patch_generation_done.value == 0 and timeout < 100:
        await tick()
        timeout += 1
    
    assert dut.patch_generation_done.value == 1, "First patch should be ready"
    
    max_patches = 1400  # Safety limit (slightly more than 1352)
    
    # Process all patches with detailed logging
    while dut.layer_processing_complete.value == 0 and patch_count < max_patches:
        # Get current state
        current_channel = int(dut.current_channel_being_processed.value)
        current_row = int(dut.current_patch_row.value)
        current_col = int(dut.current_patch_col.value)
        all_channels_done = bool(dut.all_channels_done_for_position.value)
        
        # Track this position and channel
        pos_key = (current_row, current_col)
        if pos_key not in position_channel_map:
            position_channel_map[pos_key] = []
        position_channel_map[pos_key].append(current_channel)
        
        # Get some patch data for verification
        try:
            A0_data = [dut.A0[i].value.integer for i in range(4)] if hasattr(dut, 'A0') else "N/A"
        except:
            A0_data = "N/A"
        
        patch_count += 1
        
        # Log every single extraction
        complete_marker = "✓" if all_channels_done else " "
        dut._log.info(f"[{patch_count:4d}] Pos({current_row:2d},{current_col:2d}) Ch={current_channel} {complete_marker} | A0_sample={A0_data}")
        
        # If this completes a position, show the channel sequence
        if all_channels_done:
            channels_for_pos = position_channel_map[pos_key]
            expected_sequence = list(range(8))  # [0,1,2,3,4,5,6,7]
            
            if channels_for_pos == expected_sequence:
                dut._log.info(f"      → POSITION ({current_row:2d},{current_col:2d}) COMPLETE: ✅ Perfect sequence {channels_for_pos}")
            else:
                dut._log.info(f"      → POSITION ({current_row:2d},{current_col:2d}) COMPLETE: ❌ Wrong sequence {channels_for_pos}")
                dut._log.info(f"        Expected: {expected_sequence}")
            dut._log.info("")  # Blank line for readability
        
        # Advance to next patch/channel
        dut.advance_to_next_patch.value = 1
        await tick()
        dut.advance_to_next_patch.value = 0
        
        # Start next patch generation
        dut.start_patch_generation.value = 1
        await tick()
        dut.start_patch_generation.value = 0
        
        # Wait for next patch
        timeout = 0
        while dut.patch_generation_done.value == 0 and timeout < 100:
            await tick()
            timeout += 1
        
        if timeout >= 100:
            dut._log.info(f"[{patch_count:4d}] TIMEOUT - Processing complete")
            break
    
    # Final analysis
    dut._log.info("=" * 80)
    dut._log.info("=== FINAL ANALYSIS ===")
    dut._log.info(f"Total patches processed: {patch_count}")
    dut._log.info(f"Total positions processed: {len(position_channel_map)}")
    
    # Analyze each position
    perfect_positions = 0
    error_positions = 0
    
    dut._log.info("\n=== POSITION-BY-POSITION CHANNEL ANALYSIS ===")
    
    for row in range(13):  # 0 to 12 (13 rows)
        for col in range(13):  # 0 to 12 (13 cols)
            pos_key = (row, col)
            if pos_key in position_channel_map:
                channels = position_channel_map[pos_key]
                expected = list(range(8))
                
                if channels == expected:
                    status = "✅"
                    perfect_positions += 1
                else:
                    status = "❌"
                    error_positions += 1
                    
                dut._log.info(f"Pos({row:2d},{col:2d}): {channels} {status}")
                
                if channels != expected:
                    dut._log.info(f"    Expected: {expected}")
                    dut._log.info(f"    Missing: {set(expected) - set(channels)}")
                    dut._log.info(f"    Extra: {set(channels) - set(expected)}")
            else:
                dut._log.info(f"Pos({row:2d},{col:2d}): NOT PROCESSED ❌")
                error_positions += 1
    
    # Summary
    dut._log.info("\n=== SUMMARY ===")
    dut._log.info(f"Perfect positions (all 8 channels): {perfect_positions}")
    dut._log.info(f"Error positions: {error_positions}")
    dut._log.info(f"Expected total positions: 169 (13x13)")
    
    if perfect_positions == 169 and error_positions == 0:
        dut._log.info("🎉 SUCCESS: All positions processed perfectly with all 8 channels!")
    else:
        dut._log.info("❌ ISSUES DETECTED: Some positions missing or have wrong channels")
    
    # Channel distribution check
    all_channels = []
    for channels in position_channel_map.values():
        all_channels.extend(channels)
    
    channel_counts = {i: all_channels.count(i) for i in range(8)}
    dut._log.info("\nChannel distribution:")
    for ch in range(8):
        dut._log.info(f"  Channel {ch}: {channel_counts[ch]} times")
    
    expected_per_channel = 169  # Should be 169 times each
    all_equal = all(count == expected_per_channel for count in channel_counts.values())
    
    if all_equal:
        dut._log.info("✅ Perfect channel distribution: each channel appears exactly 169 times")
    else:
        dut._log.info("❌ Uneven channel distribution detected")
    
    dut._log.info("=== DETAILED TRACE COMPLETE ===") 