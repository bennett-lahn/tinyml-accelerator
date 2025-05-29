#!/usr/bin/env python3

import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, FallingEdge, Timer
from cocotb.result import TestFailure
import struct

def convert_to_signed(val, bits):
    """Convert unsigned value to signed"""
    if val & (1 << (bits-1)):
        return val - (1 << bits)
    return val

def extract_pixels_from_word(word_val):
    """Extract 4 pixels from a 32-bit word (MSB to LSB)"""
    pixels = []
    for i in range(4):
        pixel = (word_val >> (24 - i*8)) & 0xFF
        pixels.append(convert_to_signed(pixel, 8))
    return pixels

@cocotb.test()
async def test_patch_extractor_only(dut):
    """Test the patch extractor directly with zero padding"""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    # Configure for 16x16x8 input
    dut.img_width.value = 16
    dut.img_height.value = 16
    dut.num_channels.value = 8
    await RisingEdge(dut.clk)
    
    print("=== Testing Patch Extractor Zero Padding ===")
    print("Input: 16x16x8 image with SAME padding")
    print("Patch size: 4x4, PAD_LEFT=1, PAD_RIGHT=2, PAD_TOP=1, PAD_BOTTOM=2")
    print("")
    
    # Start extraction
    dut.start_patch_extraction.value = 1
    await RisingEdge(dut.clk)
    dut.start_patch_extraction.value = 0
    
    patch_count = 0
    
    # Monitor patches, focusing on boundary and interior positions
    for cycle in range(3000):  # Increase cycles to reach interior positions
        await RisingEdge(dut.clk)
        
        # Check if patch extractor has valid output
        if dut.patch_valid.value == 1:
            row = int(dut.current_patch_row.value)
            col = int(dut.current_patch_col.value)
            channel = int(dut.current_channel.value)
            
            # Only print details for specific test cases
            if (row == 0 and col == 0 and channel == 0) or \
               (row == 1 and col == 1 and channel == 0) or \
               (row == 2 and col == 2 and channel == 0) or \
               (channel == 0 and row < 3 and col < 3):
                
                print(f"\n--- PATCH {patch_count} (Position: {row}, {col}, Channel: {channel}) ---")
                
                # Get patch extractor outputs directly
                patch_A0 = int(dut.patch_A0_out.value)
                patch_A1 = int(dut.patch_A1_out.value) 
                patch_A2 = int(dut.patch_A2_out.value)
                patch_A3 = int(dut.patch_A3_out.value)
                
                print(f"A0 (row {row-1}): 0x{patch_A0:08x} = {extract_pixels_from_word(patch_A0)}")
                print(f"A1 (row {row+0}): 0x{patch_A1:08x} = {extract_pixels_from_word(patch_A1)}")
                print(f"A2 (row {row+1}): 0x{patch_A2:08x} = {extract_pixels_from_word(patch_A2)}")
                print(f"A3 (row {row+2}): 0x{patch_A3:08x} = {extract_pixels_from_word(patch_A3)}")
                
                # Analyze the padding for this position
                patch_window_top = row - 1  # PAD_TOP = 1
                patch_window_left = col - 1  # PAD_LEFT = 1
                
                print(f"Expected padding analysis:")
                print(f"  - Window spans image rows {patch_window_top} to {patch_window_top+3}")
                print(f"  - Window spans image cols {patch_window_left} to {patch_window_left+3}")
                
                # Check A0 specifically (the problematic row)
                A0_pixels = extract_pixels_from_word(patch_A0)
                expected_A0_zeros = patch_window_top < 0  # Row -1 is padding
                
                if expected_A0_zeros and all(pixel == 0 for pixel in A0_pixels):
                    print(f"✅ A0 correctly all zeros (padding row)")
                elif not expected_A0_zeros and any(pixel != 0 for pixel in A0_pixels):
                    print(f"✅ A0 has real data as expected")
                elif expected_A0_zeros and any(pixel != 0 for pixel in A0_pixels):
                    print(f"❌ A0 should be padding but has data: {A0_pixels}")
                else:
                    print(f"❌ A0 should have real data but is all zeros: {A0_pixels}")
            
            # Request next patch
            dut.next_patch.value = 1
            await RisingEdge(dut.clk)
            dut.next_patch.value = 0
            
            patch_count += 1
            
            # Stop after checking several positions to see the pattern
            if patch_count >= 100:  # Check more patches
                break
    
    print(f"\nProcessed {patch_count} patches")
    print("=== Test Complete ===") 