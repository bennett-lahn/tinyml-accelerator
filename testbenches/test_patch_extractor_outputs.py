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
async def test_patch_extractor_outputs(dut):
    """Test what the patch extractor is outputting"""
    
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
    dut.current_img_width.value = 16
    dut.current_img_height.value = 16
    dut.current_num_channels.value = 8
    await RisingEdge(dut.clk)
    
    print("=== Testing Patch Extractor Direct Outputs ===")
    print("Input: 16x16x8 image with SAME padding")
    print("Patch size: 4x4, Expected output patches: 16x16")
    print("")
    
    # Start extraction
    dut.start_patch_generation.value = 1
    await RisingEdge(dut.clk)
    dut.start_patch_generation.value = 0
    
    patch_count = 0
    
    # Monitor patches from patch extractor outputs, especially looking for interior positions
    for cycle in range(2000):  # Limit cycles to prevent infinite loop
        await RisingEdge(dut.clk)
        
        # Check if patch extractor has valid output
        if dut.u_patch_extractor.patch_valid.value == 1:
            row = int(dut.u_patch_extractor.current_patch_row.value)
            col = int(dut.u_patch_extractor.current_patch_col.value)
            channel = int(dut.u_patch_extractor.current_channel.value)
            
            # Only print details for specific test cases - focus on interior positions
            if (row == 0 and col == 0 and channel == 0) or \
               (row == 1 and col == 1 and channel == 0) or \
               (row == 2 and col == 2 and channel == 0) or \
               (row == 3 and col == 3 and channel == 0) or \
               (channel == 0 and row <= 4 and col <= 4):
                
                print(f"\n--- PATCH {patch_count} (Position: {row}, {col}, Channel: {channel}) ---")
                
                # Get patch extractor outputs directly
                patch_A0 = int(dut.u_patch_extractor.patch_A0_out.value)
                patch_A1 = int(dut.u_patch_extractor.patch_A1_out.value) 
                patch_A2 = int(dut.u_patch_extractor.patch_A2_out.value)
                patch_A3 = int(dut.u_patch_extractor.patch_A3_out.value)
                
                print(f"Patch Extractor Output A0: 0x{patch_A0:08x} = {extract_pixels_from_word(patch_A0)}")
                print(f"Patch Extractor Output A1: 0x{patch_A1:08x} = {extract_pixels_from_word(patch_A1)}")
                print(f"Patch Extractor Output A2: 0x{patch_A2:08x} = {extract_pixels_from_word(patch_A2)}")
                print(f"Patch Extractor Output A3: 0x{patch_A3:08x} = {extract_pixels_from_word(patch_A3)}")
                
                # Check if A0 should have real data (interior positions where row >= 1)
                if row >= 1:
                    A0_pixels = extract_pixels_from_word(patch_A0)
                    if all(pixel == 0 for pixel in A0_pixels):
                        print(f"⚠️  A0 is all zeros at interior position ({row},{col}) - this may indicate a problem!")
                    else:
                        print(f"✅ A0 has real data at interior position ({row},{col})")
                else:
                    print(f"✅ A0 correctly all zeros at boundary position ({row},{col})")
            
            # Request next patch
            dut.advance_to_next_patch.value = 1
            await RisingEdge(dut.clk)
            dut.advance_to_next_patch.value = 0
            
            patch_count += 1
            
            # Stop after checking enough patches to see interior positions
            if patch_count >= 200:  # Need more patches to reach row 1 positions (16*8 = 128 patches for row 0)
                break
    
    print(f"\nProcessed {patch_count} patches")
    print("=== Test Complete ===") 