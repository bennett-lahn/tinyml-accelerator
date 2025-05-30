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
async def test_patch_extractor_direct(dut):
    """Test patch extractor with padding - direct outputs only"""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    # Configure for 32x32x1 input
    dut.current_img_width.value = 32
    dut.current_img_height.value = 32
    dut.current_num_channels.value = 1
    await RisingEdge(dut.clk)
    
    print("=== Testing Patch Extractor with Zero Padding ===")
    print("Input: 32x32x1 image with SAME padding")
    print("Patch size: 4x4, PAD_LEFT=1, PAD_RIGHT=2, PAD_TOP=1, PAD_BOTTOM=2")
    print("Expected output grid: 32x32 patches")
    print("Focus: Direct patch extractor outputs (bypassing sliding window)")
    print("")
    
    # Start extraction
    dut.start_patch_generation.value = 1
    await RisingEdge(dut.clk)
    dut.start_patch_generation.value = 0
    
    patch_count = 0
    patches_per_row = 32
    
    # Track what we've seen
    boundary_patches = []
    interior_patches = []
    
    # Monitor patches from patch extractor - show key positions
    for cycle in range(5000):  # Enough cycles to reach interior positions
        await RisingEdge(dut.clk)
        
        # Check if patch extractor has valid output
        if dut.u_patch_extractor.patch_valid.value == 1:
            row = int(dut.u_patch_extractor.current_patch_row.value)
            col = int(dut.u_patch_extractor.current_patch_col.value)
            channel = int(dut.u_patch_extractor.current_channel.value)
            
            # Get patch extractor outputs directly
            patch_A0 = int(dut.u_patch_extractor.patch_A0_out.value)
            patch_A1 = int(dut.u_patch_extractor.patch_A1_out.value) 
            patch_A2 = int(dut.u_patch_extractor.patch_A2_out.value)
            patch_A3 = int(dut.u_patch_extractor.patch_A3_out.value)
            
            # Show detailed output for specific positions
            show_details = False
            
            # Show boundary positions (row 0) and interior positions (row 1, 2, 3)
            if channel == 0:  # Only show channel 0 for clarity
                if row == 0 and col <= 3:  # First few boundary positions
                    show_details = True
                    boundary_patches.append((row, col))
                elif row >= 1 and row <= 3 and col <= 3:  # Interior positions
                    show_details = True
                    interior_patches.append((row, col))
            
            if show_details:
                print(f"\n--- PATCH {patch_count} (Position: {row}, {col}, Channel: {channel}) ---")
                
                A0_pixels = extract_pixels_from_word(patch_A0)
                A1_pixels = extract_pixels_from_word(patch_A1)
                A2_pixels = extract_pixels_from_word(patch_A2)
                A3_pixels = extract_pixels_from_word(patch_A3)
                
                print(f"A0 (img row {row-1}): {A0_pixels}")
                print(f"A1 (img row {row+0}): {A1_pixels}")
                print(f"A2 (img row {row+1}): {A2_pixels}")
                print(f"A3 (img row {row+2}): {A3_pixels}")
                
                # Analyze padding behavior
                patch_window_top = row - 1  # Due to PAD_TOP = 1
                
                if row == 0:
                    print(f"✅ Boundary position: A0 represents image row {patch_window_top} (padding)")
                    if all(pixel == 0 for pixel in A0_pixels):
                        print(f"✅ A0 correctly all zeros (padding)")
                    else:
                        print(f"❌ A0 should be zeros but has: {A0_pixels}")
                else:
                    print(f"📍 Interior position: A0 represents image row {patch_window_top} (real data)")
                    if all(pixel == 0 for pixel in A0_pixels):
                        print(f"❌ A0 is all zeros but should have real data!")
                    else:
                        print(f"✅ A0 has real data: {A0_pixels}")
            
            # Request next patch
            dut.advance_to_next_patch.value = 1
            await RisingEdge(dut.clk)
            dut.advance_to_next_patch.value = 0
            
            patch_count += 1
            
            # Stop after we've seen enough examples
            if len(boundary_patches) >= 4 and len(interior_patches) >= 9:
                break
    
    print(f"\n=== SUMMARY ===")
    print(f"Total patches processed: {patch_count}")
    print(f"Boundary positions tested: {boundary_patches}")
    print(f"Interior positions tested: {interior_patches}")
    print("=== Test Complete ===") 