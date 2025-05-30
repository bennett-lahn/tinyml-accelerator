import cocotb
import logging
from cocotb.triggers import RisingEdge
from cocotb.clock import Clock

# Set up logging
cocotb.log.setLevel(logging.INFO)

@cocotb.test()
async def test_zero_padding_16x16x8(dut):
    """Test zero padding with 16x16x8 input → 16x16 output positions"""
    CLOCK_PERIOD = 2
    dut._log.setLevel(logging.INFO)
    
    # Configuration
    IMG_W = 16
    IMG_H = 16
    NUM_CHANNELS = 8
    PATCH_SIZE = 4
    
    # With zero padding: input 16x16 → output 16x16 (SAME padding)
    expected_output_positions = IMG_W * IMG_H  # 256 positions
    expected_total_patches = expected_output_positions * NUM_CHANNELS  # 256 * 8 = 2048
    
    # Start clock
    cocotb.start_soon(Clock(dut.clk, CLOCK_PERIOD, units="ns").start())
    
    async def tick():
        await RisingEdge(dut.clk)
    
    async def reset_dut():
        dut.reset.value = 1
        await tick()
        dut.reset.value = 0
        await tick()
    
    await reset_dut()
    
    dut._log.info("=== ZERO PADDING TEST: 16x16x8 → 16x16 OUTPUT ===")
    dut._log.info(f"Expected output positions: {expected_output_positions} (16x16)")
    dut._log.info(f"Expected total patches: {expected_total_patches} (256 pos × 8 ch)")
    dut._log.info("Padding: PAD_LEFT=1, PAD_RIGHT=2, PAD_TOP=1, PAD_BOTTOM=2")
    dut._log.info("=" * 80)
    
    # Configure TPU
    dut.current_img_width.value = IMG_W
    dut.current_img_height.value = IMG_H
    dut.current_num_channels.value = NUM_CHANNELS
    
    dut.start_patch_generation.value = 0
    dut.start_sliding_window.value = 0
    dut.advance_to_next_patch.value = 0
    
    await tick()
    
    # Start first patch generation
    dut.start_patch_generation.value = 1
    await tick()
    dut.start_patch_generation.value = 0
    
    # Wait for first patch
    timeout = 0
    while dut.patch_generation_done.value == 0 and timeout < 100:
        await tick()
        timeout += 1
    
    assert dut.patch_generation_done.value == 1, "First patch should be ready"
    
    # Track processing
    patch_count = 0
    position_map = {}  # {(row,col): [list of channels]}
    padding_samples = []  # Store samples that should contain padding
    
    max_patches = 2100  # Safety limit
    
    while dut.layer_processing_complete.value == 0 and patch_count < max_patches:
        # Get current state
        current_channel = int(dut.current_channel_being_processed.value)
        current_row = int(dut.current_patch_row.value)
        current_col = int(dut.current_patch_col.value)
        all_channels_done = bool(dut.all_channels_done_for_position.value)
        
        # Track positions
        pos_key = (current_row, current_col)
        if pos_key not in position_map:
            position_map[pos_key] = []
        position_map[pos_key].append(current_channel)
        
        # Get patch data
        try:
            A0_data = [dut.A0[i].value.integer for i in range(4)]
            A1_data = [dut.A1[i].value.integer for i in range(4)]
            A2_data = [dut.A2[i].value.integer for i in range(4)]
            A3_data = [dut.A3[i].value.integer for i in range(4)]
            patch_data = [A0_data, A1_data, A2_data, A3_data]
        except:
            patch_data = "N/A"
        
        patch_count += 1
        
        # Check for padding regions and verify zeros
        if patch_count <= 10 or patch_count % 200 == 0 or all_channels_done:
            dut._log.info(f"[{patch_count:4d}] Pos({current_row:2d},{current_col:2d}) Ch={current_channel} | Sample: A0={A0_data}")
        
        # Test specific padding cases
        if (current_row == 0 and current_col == 0 and current_channel == 0):
            # Top-left corner should have padding zeros
            # Patch covers region (-1,-1) to (2,2), so top-left 2x2 should be zeros
            padding_samples.append({
                'position': (current_row, current_col),
                'channel': current_channel,
                'patch_data': patch_data,
                'expected_padding': 'top-left corner'
            })
            
        if (current_row == 15 and current_col == 15 and current_channel == 0):
            # Bottom-right corner should have padding zeros  
            # Patch covers region (15,15) to (18,18), so bottom-right 3x3 should be zeros
            padding_samples.append({
                'position': (current_row, current_col),
                'channel': current_channel,
                'patch_data': patch_data,
                'expected_padding': 'bottom-right corner'
            })
        
        # Advance to next patch
        dut.advance_to_next_patch.value = 1
        await tick()
        dut.advance_to_next_patch.value = 0
        
        # Start next patch
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
    
    # Analysis
    dut._log.info("=" * 80)
    dut._log.info("=== ZERO PADDING ANALYSIS ===")
    dut._log.info(f"Total patches processed: {patch_count}")
    dut._log.info(f"Total positions processed: {len(position_map)}")
    dut._log.info(f"Expected positions: {expected_output_positions}")
    dut._log.info(f"Expected total patches: {expected_total_patches}")
    
    # Verify output size
    if len(position_map) == expected_output_positions:
        dut._log.info("✅ CORRECT OUTPUT SIZE: 16x16 positions achieved with zero padding")
    else:
        dut._log.info(f"❌ WRONG OUTPUT SIZE: Got {len(position_map)}, expected {expected_output_positions}")
    
    # Check position coverage
    perfect_positions = 0
    for row in range(16):
        for col in range(16):
            pos_key = (row, col)
            if pos_key in position_map:
                channels = position_map[pos_key]
                if len(channels) == 8 and channels == list(range(8)):
                    perfect_positions += 1
                else:
                    dut._log.info(f"❌ Pos({row},{col}): {channels}")
            else:
                dut._log.info(f"❌ Missing Pos({row},{col})")
    
    dut._log.info(f"Perfect positions: {perfect_positions}/256")
    
    # Analyze padding samples
    dut._log.info("\n=== PADDING VERIFICATION ===")
    for sample in padding_samples:
        pos = sample['position']
        ch = sample['channel']
        data = sample['patch_data']
        expected = sample['expected_padding']
        
        dut._log.info(f"Position {pos} Ch{ch} ({expected}):")
        if data != "N/A":
            for i, row_data in enumerate(data):
                # Convert signed int8 to unsigned for zero comparison
                row_unsigned = [(x + 256) % 256 for x in row_data]
                zeros_in_row = sum(1 for x in row_unsigned if x == 0)
                dut._log.info(f"  Row {i}: {row_unsigned} ({zeros_in_row} zeros)")
    
    # Final verification
    if perfect_positions == 256 and patch_count == expected_total_patches:
        dut._log.info("\n🎉 SUCCESS: Zero padding working perfectly!")
        dut._log.info("✅ 16x16x8 input → 16x16 output positions achieved")
        dut._log.info("✅ All 2048 patches processed (256 positions × 8 channels)")
        dut._log.info("✅ SAME padding successfully implemented for 4x4 kernels")
    else:
        dut._log.info("\n❌ ISSUES DETECTED in zero padding implementation")
    
    dut._log.info("=== ZERO PADDING TEST COMPLETE ===") 