import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
from cocotb.binary import BinaryValue
import numpy as np

@cocotb.test()
async def test_deep_layer_4x4x32_detailed(dut):
    """Detailed analysis of 4x4x32 deep layer processing"""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    cocotb.log.info("=== DEEP LAYER ANALYSIS: 4x4x32 ===")
    
    # Reset
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    # Configure for 4x4x32 (deep layer)
    img_width = 4
    img_height = 4
    num_channels = 32
    
    dut.img_width.value = img_width
    dut.img_height.value = img_height
    dut.num_channels.value = num_channels
    dut.pad_top.value = 0
    dut.pad_bottom.value = 0
    dut.pad_left.value = 0
    dut.pad_right.value = 0
    
    channel_groups = (num_channels + 3) // 4  # Should be 8
    cocotb.log.info(f"Image: {img_height}x{img_width}x{num_channels}")
    cocotb.log.info(f"Channel groups: {channel_groups}")
    cocotb.log.info(f"Expected spatial blocks: 1x1 = 1 (since 4x4 fits in one block)")
    cocotb.log.info("")
    
    # Start extraction
    dut.start_extraction.value = 1
    await RisingEdge(dut.clk)
    dut.start_extraction.value = 0
    
    # Test all channel groups for the single spatial block
    for cg in range(channel_groups):
        cocotb.log.info(f"--- Channel Group {cg}: channels {cg*4}-{cg*4+3} ---")
        
        # Monitor RAM accesses during loading
        ram_accesses = []
        loading_cycles = 0
        
        while True:
            await RisingEdge(dut.clk)
            loading_cycles += 1
            
            # Track RAM accesses
            if dut.unified_buffer_inst.ram_re.value == 1:
                addr = int(dut.unified_buffer_inst.ram_addr.value)
                ram_accesses.append(addr)
                
                # Log RAM data
                ram_data = {
                    'dout0': int(dut.ram_dout0.value),
                    'dout1': int(dut.ram_dout1.value),
                    'dout2': int(dut.ram_dout2.value),
                    'dout3': int(dut.ram_dout3.value)
                }
                
                cocotb.log.info(f"  RAM[{addr}]: dout3=0x{ram_data['dout3']:08x} dout2=0x{ram_data['dout2']:08x} dout1=0x{ram_data['dout1']:08x} dout0=0x{ram_data['dout0']:08x}")
            
            if dut.buffer_loading_complete.value == 1:
                break
                
            if loading_cycles > 300:
                cocotb.log.error(f"Timeout waiting for channel group {cg}")
                break
        
        cocotb.log.info(f"  Loading completed in {loading_cycles} cycles")
        cocotb.log.info(f"  RAM addresses accessed: {ram_accesses}")
        
        # Check the 7x7 buffer outputs - focus on the valid 4x4 region
        cocotb.log.info("  7x7 Buffer contents (showing valid 4x4 region):")
        
        # The valid 4x4 data should be in positions [1:5, 1:5] of the 7x7 buffer
        # (assuming no padding, so the 4x4 image starts at buffer position (1,1))
        valid_positions = [
            (dut.patch_pe11_out, "(1,1)"), (dut.patch_pe12_out, "(1,2)"), (dut.patch_pe13_out, "(1,3)"), (dut.patch_pe14_out, "(1,4)"),
            (dut.patch_pe21_out, "(2,1)"), (dut.patch_pe22_out, "(2,2)"), (dut.patch_pe23_out, "(2,3)"), (dut.patch_pe24_out, "(2,4)"),
            (dut.patch_pe31_out, "(3,1)"), (dut.patch_pe32_out, "(3,2)"), (dut.patch_pe33_out, "(3,3)"), (dut.patch_pe34_out, "(3,4)"),
            (dut.patch_pe41_out, "(4,1)"), (dut.patch_pe42_out, "(4,2)"), (dut.patch_pe43_out, "(4,3)"), (dut.patch_pe44_out, "(4,4)")
        ]
        
        for i, (patch_signal, pos) in enumerate(valid_positions):
            patch_data = int(patch_signal.value)
            channels = [
                (patch_data >> 0) & 0xFF,
                (patch_data >> 8) & 0xFF,
                (patch_data >> 16) & 0xFF,
                (patch_data >> 24) & 0xFF
            ]
            cocotb.log.info(f"    {pos}: 0x{patch_data:08x} -> channels {channels}")
        
        # Move to next channel group (except for last one)
        if cg < channel_groups - 1:
            dut.next_channel_group.value = 1
            await RisingEdge(dut.clk)
            dut.next_channel_group.value = 0
            await RisingEdge(dut.clk)
            await RisingEdge(dut.clk)
        
        cocotb.log.info("")
    
    cocotb.log.info("=== DEEP LAYER ANALYSIS COMPLETE ===")

@cocotb.test()
async def test_same_padding_behavior(dut):
    """Test 'same' padding behavior for different layer sizes"""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    cocotb.log.info("=== TESTING 'SAME' PADDING BEHAVIOR ===")
    cocotb.log.info("Testing padding that would be used for 'same' convolution with 4x4 kernels")
    cocotb.log.info("")
    
    # For 4x4 kernels with 'same' padding:
    # - Padding needed = (kernel_size - 1) / 2 = (4-1)/2 = 1.5
    # - Typically implemented as: pad_left=1, pad_right=2, pad_top=1, pad_bottom=2
    # - Or symmetric: pad=1 on all sides (losing 1 pixel on right/bottom)
    
    same_padding_cases = [
        # (height, width, channels, pad_top, pad_bottom, pad_left, pad_right, description)
        (32, 32, 1, 1, 2, 1, 2, "32x32 with asymmetric same padding"),
        (32, 32, 1, 1, 1, 1, 1, "32x32 with symmetric same padding"),
        (16, 16, 8, 1, 2, 1, 2, "16x16x8 with asymmetric same padding"),
        (8, 8, 16, 1, 2, 1, 2, "8x8x16 with asymmetric same padding"),
        (4, 4, 32, 1, 2, 1, 2, "4x4x32 with asymmetric same padding"),
        (4, 4, 32, 2, 2, 2, 2, "4x4x32 with extra padding"),
    ]
    
    for height, width, channels, pad_top, pad_bottom, pad_left, pad_right, description in same_padding_cases:
        cocotb.log.info(f"=== Testing: {description} ===")
        
        # Reset
        dut.reset.value = 1
        await RisingEdge(dut.clk)
        await RisingEdge(dut.clk)
        dut.reset.value = 0
        await RisingEdge(dut.clk)
        
        # Configure
        dut.img_width.value = width
        dut.img_height.value = height
        dut.num_channels.value = channels
        dut.pad_top.value = pad_top
        dut.pad_bottom.value = pad_bottom
        dut.pad_left.value = pad_left
        dut.pad_right.value = pad_right
        
        # Calculate dimensions
        padded_width = width + pad_left + pad_right
        padded_height = height + pad_top + pad_bottom
        channel_groups = (channels + 3) // 4
        spatial_blocks_x = (padded_width + 3) // 4  # Based on padded dimensions
        spatial_blocks_y = (padded_height + 3) // 4
        
        cocotb.log.info(f"  Original: {height}x{width}x{channels}")
        cocotb.log.info(f"  Padding: top={pad_top}, bottom={pad_bottom}, left={pad_left}, right={pad_right}")
        cocotb.log.info(f"  Padded: {padded_height}x{padded_width}")
        cocotb.log.info(f"  Expected spatial blocks: {spatial_blocks_y}x{spatial_blocks_x} = {spatial_blocks_y * spatial_blocks_x}")
        cocotb.log.info(f"  Channel groups: {channel_groups}")
        
        # Start extraction
        dut.start_extraction.value = 1
        await RisingEdge(dut.clk)
        dut.start_extraction.value = 0
        
        # Test first channel group of first spatial block
        loading_cycles = 0
        success = False
        
        while loading_cycles < 500:
            await RisingEdge(dut.clk)
            loading_cycles += 1
            
            if dut.buffer_loading_complete.value == 1:
                success = True
                break
        
        if success:
            cocotb.log.info(f"  ✓ PASSED - Loaded in {loading_cycles} cycles")
            
            # Check corner positions to see padding behavior
            corner_positions = [
                (dut.patch_pe00_out, "(0,0) - top-left corner"),
                (dut.patch_pe06_out, "(0,6) - top-right corner"),
                (dut.patch_pe60_out, "(6,0) - bottom-left corner"),
                (dut.patch_pe66_out, "(6,6) - bottom-right corner"),
                (dut.patch_pe33_out, "(3,3) - center")
            ]
            
            cocotb.log.info("  Buffer corner analysis:")
            for patch_signal, description in corner_positions:
                patch_data = int(patch_signal.value)
                cocotb.log.info(f"    {description}: 0x{patch_data:08x}")
        else:
            cocotb.log.error(f"  ✗ FAILED - Timeout after {loading_cycles} cycles")
        
        cocotb.log.info("")
    
    cocotb.log.info("=== SAME PADDING TEST COMPLETE ===")

@cocotb.test()
async def test_spatial_block_calculation(dut):
    """Test how spatial blocks are calculated for different input sizes"""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    cocotb.log.info("=== SPATIAL BLOCK CALCULATION TEST ===")
    
    test_cases = [
        # (height, width, channels, expected_blocks_y, expected_blocks_x, description)
        (4, 4, 32, 1, 1, "Exact fit: 4x4 -> 1 block"),
        (8, 8, 16, 2, 2, "Exact fit: 8x8 -> 4 blocks"),
        (5, 5, 8, 2, 2, "Partial fit: 5x5 -> 4 blocks"),
        (7, 7, 4, 2, 2, "Partial fit: 7x7 -> 4 blocks"),
        (3, 3, 1, 1, 1, "Small: 3x3 -> 1 block"),
        (1, 1, 1, 1, 1, "Minimum: 1x1 -> 1 block"),
    ]
    
    for height, width, channels, expected_blocks_y, expected_blocks_x, description in test_cases:
        cocotb.log.info(f"Testing: {description}")
        
        # Reset
        dut.reset.value = 1
        await RisingEdge(dut.clk)
        await RisingEdge(dut.clk)
        dut.reset.value = 0
        await RisingEdge(dut.clk)
        
        # Configure
        dut.img_width.value = width
        dut.img_height.value = height
        dut.num_channels.value = channels
        dut.pad_top.value = 0
        dut.pad_bottom.value = 0
        dut.pad_left.value = 0
        dut.pad_right.value = 0
        
        # Calculate actual spatial blocks
        actual_blocks_x = (width + 3) // 4
        actual_blocks_y = (height + 3) // 4
        
        cocotb.log.info(f"  Input: {height}x{width}x{channels}")
        cocotb.log.info(f"  Expected blocks: {expected_blocks_y}x{expected_blocks_x}")
        cocotb.log.info(f"  Calculated blocks: {actual_blocks_y}x{actual_blocks_x}")
        
        if actual_blocks_x == expected_blocks_x and actual_blocks_y == expected_blocks_y:
            cocotb.log.info(f"  ✓ Block calculation correct")
        else:
            cocotb.log.error(f"  ✗ Block calculation mismatch")
        
        # Test that first spatial block loads successfully
        dut.start_extraction.value = 1
        await RisingEdge(dut.clk)
        dut.start_extraction.value = 0
        
        loading_cycles = 0
        success = False
        
        while loading_cycles < 300:
            await RisingEdge(dut.clk)
            loading_cycles += 1
            
            if dut.buffer_loading_complete.value == 1:
                success = True
                break
        
        if success:
            cocotb.log.info(f"  ✓ First spatial block loaded successfully ({loading_cycles} cycles)")
        else:
            cocotb.log.error(f"  ✗ First spatial block failed to load")
        
        cocotb.log.info("")
    
    cocotb.log.info("=== SPATIAL BLOCK CALCULATION TEST COMPLETE ===") 