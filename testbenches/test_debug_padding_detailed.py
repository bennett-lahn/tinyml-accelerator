import cocotb
from cocotb.clock import Clock
from cocotb.triggers import Timer, RisingEdge, FallingEdge
from cocotb.result import TestFailure
import random

def signed_to_int(value, bits):
    """Convert signed binary to integer"""
    if value & (1 << (bits-1)):
        return value - (1 << bits)
    return value

def int_to_signed(value, bits):
    """Convert integer to signed binary"""
    if value < 0:
        return (1 << bits) + value
    return value

@cocotb.test()
async def debug_detailed_padding_logic(dut):
    """Debug detailed padding logic with signal tracing"""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset.value = 1
    dut.start_patch_generation.value = 0
    dut.advance_to_next_patch.value = 0
    
    # Wait for reset
    await Timer(20, units="ns")
    dut.reset.value = 0
    await Timer(20, units="ns")
    
    # Configure for 16x16x8 processing
    dut.current_img_width.value = 16
    dut.current_img_height.value = 16
    dut.current_num_channels.value = 8
    
    # Start processing
    dut.start_patch_generation.value = 1
    await RisingEdge(dut.clk)
    dut.start_patch_generation.value = 0
    
    dut._log.info("=== DETAILED PADDING DEBUG ===")
    dut._log.info("Looking for Position (1,1) Channel 0 - should access valid pixel (0,0)")
    
    # Look for the first few patches and examine signals in detail
    patches_analyzed = 0
    max_patches_to_analyze = 10
    
    while patches_analyzed < max_patches_to_analyze:
        # Wait for patch_ready
        while dut.patch_generation_done.value == 0:
            await RisingEdge(dut.clk)
            
            # Examine critical signals during FETCH_PIXELS state
            if dut.u_patch_extractor.current_state.value == 1:  # FETCH_PIXELS = 1
                patch_row = int(dut.u_patch_extractor.patch_row.value)
                patch_col = int(dut.u_patch_extractor.patch_col.value)
                patch_channel = int(dut.u_patch_extractor.patch_channel.value)
                row_offset = int(dut.u_patch_extractor.current_patch_row_offset.value)
                col_offset = int(dut.u_patch_extractor.pixels_fetched_in_row.value)
                
                # Calculate expected absolute coordinates
                abs_row = patch_row + row_offset - 1  # PAD_TOP = 1
                abs_col = patch_col + col_offset - 1  # PAD_LEFT = 1
                
                # Read actual hardware values
                hw_abs_row = signed_to_int(int(dut.u_patch_extractor.abs_pixel_row.value), 7)
                hw_abs_col = signed_to_int(int(dut.u_patch_extractor.abs_pixel_col.value), 7)
                
                is_padding = int(dut.u_patch_extractor.is_padding_pixel.value)
                is_valid_access = int(dut.u_patch_extractor.is_valid_ram_access.value)
                ram_re = int(dut.u_patch_extractor.ram_re.value)
                
                dut._log.info(f"FETCH: Pos({patch_row},{patch_col}) Ch{patch_channel} Pixel({row_offset},{col_offset})")
                dut._log.info(f"  Expected abs: ({abs_row},{abs_col}) | HW abs: ({hw_abs_row},{hw_abs_col})")
                dut._log.info(f"  is_padding={is_padding} | is_valid_access={is_valid_access} | ram_re={ram_re}")
                
                if patch_row == 1 and patch_col == 1 and patch_channel == 0:
                    if row_offset == 0 and col_offset == 0:
                        dut._log.info(f"🎯 TARGET PIXEL FOUND: Pos(1,1) Ch0 Pixel(0,0)")
                        dut._log.info(f"  Should access valid pixel (0,0) in image")
                        dut._log.info(f"  Expected: abs_row=0, abs_col=0, padding=False, ram_re=True")
                        dut._log.info(f"  Actual:   abs_row={hw_abs_row}, abs_col={hw_abs_col}, padding={bool(is_padding)}, ram_re={bool(ram_re)}")
                        
                        if is_padding:
                            dut._log.error("❌ PADDING DETECTION ERROR: Valid pixel marked as padding!")
                        if not ram_re:
                            dut._log.error("❌ RAM_RE ERROR: Valid pixel not triggering RAM read!")
        
        # Process the ready patch
        patches_analyzed += 1
        
        patch_row = int(dut.current_patch_row.value)
        patch_col = int(dut.current_patch_col.value) 
        patch_channel = int(dut.current_channel_being_processed.value)
        
        A0 = [
            int(dut.A0[0].value),
            int(dut.A0[1].value),
            int(dut.A0[2].value),
            int(dut.A0[3].value)
        ]
        
        dut._log.info(f"[{patches_analyzed:4d}] Pos({patch_row:2d},{patch_col:2d}) Ch={patch_channel} | A0={A0}")
        
        # Special focus on position (1,1) 
        if patch_row == 1 and patch_col == 1 and patch_channel == 0:
            dut._log.info(f"🎯 ANALYZING TARGET POSITION (1,1) Ch0:")
            dut._log.info(f"  A0 row 0 (should access abs pixels (0,0)-(0,3)): {A0}")
            if all(x == 0 for x in A0):
                dut._log.error("❌ ALL ZEROS - Position (1,1) Ch0 should have non-zero data!")
                
                # Get more debug info about the RAM access
                word_addr = int(dut.u_patch_extractor.word_addr.value)
                pixel_offset = int(dut.u_patch_extractor.pixel_offset_in_word.value)
                pixel_base_addr = int(dut.u_patch_extractor.pixel_base_addr.value)
                
                dut._log.info(f"  Debug: word_addr={word_addr}, pixel_offset={pixel_offset}, pixel_base_addr={pixel_base_addr}")
            else:
                dut._log.info("✅ Found non-zero data!")
        
        # Advance to next patch
        dut.advance_to_next_patch.value = 1
        await RisingEdge(dut.clk)
        dut.advance_to_next_patch.value = 0
        
        # Wait for patch_ready to go low
        while dut.patch_generation_done.value == 1:
            await RisingEdge(dut.clk)
    
    dut._log.info("=== DETAILED PADDING DEBUG COMPLETE ===") 