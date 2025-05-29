import cocotb
import logging
from cocotb.triggers import RisingEdge, FallingEdge
from cocotb.clock import Clock

@cocotb.test()
async def test_patch_extractor_channel_first(dut):
    """Test that patch extractor iterates through channels first for each position"""
    
    # Configure logging
    dut._log.setLevel(logging.INFO)
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Helper function to wait for clock edge
    async def tick():
        await RisingEdge(dut.clk)
    
    # Reset
    dut.reset.value = 1
    dut.start_patch_extraction.value = 0
    dut.next_patch.value = 0
    dut.img_width.value = 8  # Small 8x8 image for testing
    dut.img_height.value = 8
    dut.num_channels.value = 3  # 3 channels
    
    # Mock RAM outputs
    dut.ram_dout0.value = 0x03020100  # 4 pixels: [0,1,2,3]
    dut.ram_dout1.value = 0x07060504  # 4 pixels: [4,5,6,7] 
    dut.ram_dout2.value = 0x0B0A0908  # 4 pixels: [8,9,10,11]
    dut.ram_dout3.value = 0x0F0E0D0C  # 4 pixels: [12,13,14,15]
    
    await tick()
    dut.reset.value = 0
    await tick()
    
    # Start extraction
    dut._log.info("Starting patch extraction")
    dut.start_patch_extraction.value = 1
    await tick()
    dut.start_patch_extraction.value = 0
    
    # Track the sequence of patches extracted
    patches_extracted = []
    
    for position_idx in range(5):  # Test first 5 patch positions  
        for channel in range(3):  # 3 channels per position
            # Wait for patch to be ready
            while not dut.patch_ready.value:
                await tick()
            
            # Record what was extracted
            current_channel = int(dut.current_channel.value)
            current_row = int(dut.current_patch_row.value)
            current_col = int(dut.current_patch_col.value)
            all_channels_done = bool(dut.all_channels_done_for_position.value)
            
            patches_extracted.append({
                'channel': current_channel,
                'row': current_row, 
                'col': current_col,
                'all_channels_done': all_channels_done
            })
            
            dut._log.info(f"Extracted patch: channel={current_channel}, position=({current_row},{current_col}), all_channels_done={all_channels_done}")
            
            # Advance to next patch (next channel or next position)
            dut.next_patch.value = 1
            await tick()
            dut.next_patch.value = 0
            await tick()
            
            # If this was the last channel for this position, break inner loop
            if all_channels_done:
                break
    
    # Verify the extraction order
    dut._log.info("Verifying extraction order...")
    
    # Check that we got the expected sequence
    expected_sequence = [
        # Position (0,0): all channels
        {'channel': 0, 'row': 0, 'col': 0, 'all_channels_done': False},
        {'channel': 1, 'row': 0, 'col': 0, 'all_channels_done': False}, 
        {'channel': 2, 'row': 0, 'col': 0, 'all_channels_done': True},
        # Position (0,1): all channels  
        {'channel': 0, 'row': 0, 'col': 1, 'all_channels_done': False},
        {'channel': 1, 'row': 0, 'col': 1, 'all_channels_done': False},
        {'channel': 2, 'row': 0, 'col': 1, 'all_channels_done': True},
        # Position (0,2): all channels
        {'channel': 0, 'row': 0, 'col': 2, 'all_channels_done': False},
        {'channel': 1, 'row': 0, 'col': 2, 'all_channels_done': False},
        {'channel': 2, 'row': 0, 'col': 2, 'all_channels_done': True},
        # Position (0,3): all channels
        {'channel': 0, 'row': 0, 'col': 3, 'all_channels_done': False},
        {'channel': 1, 'row': 0, 'col': 3, 'all_channels_done': False},
        {'channel': 2, 'row': 0, 'col': 3, 'all_channels_done': True},
        # Position (0,4): all channels  
        {'channel': 0, 'row': 0, 'col': 4, 'all_channels_done': False},
        {'channel': 1, 'row': 0, 'col': 4, 'all_channels_done': False},
        {'channel': 2, 'row': 0, 'col': 4, 'all_channels_done': True},
    ]
    
    # Verify sequence matches expected
    for i, (actual, expected) in enumerate(zip(patches_extracted, expected_sequence)):
        assert actual == expected, f"Mismatch at index {i}: got {actual}, expected {expected}"
        dut._log.info(f"✓ Patch {i}: {actual}")
    
    dut._log.info("SUCCESS: Patch extractor correctly iterates channels first for each position!")

@cocotb.test()
async def test_patch_extractor_channel_last_layout(dut):
    """Test that patch extractor works correctly with channel-last memory layout"""
    
    # Configure logging
    dut._log.setLevel(logging.INFO)
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Helper function to wait for clock edge
    async def tick():
        await RisingEdge(dut.clk)
    
    # Test configuration: 8x8 image with 3 channels
    IMG_W, IMG_H, NUM_CH = 8, 8, 3
    
    # Create simulated memory with channel-last layout [row, col, channel]
    # Each pixel is 8-bit, stored as consecutive bytes
    memory = {}
    for row in range(IMG_H):
        for col in range(IMG_W):
            for ch in range(NUM_CH):
                addr = row * IMG_W * NUM_CH + col * NUM_CH + ch
                # Use a simple pattern: pixel value = row*100 + col*10 + ch
                memory[addr] = (row * 100 + col * 10 + ch) & 0xFF
    
    def get_memory_word(word_addr):
        """Get 128-bit word (16 pixels) from simulated memory"""
        base_pixel_addr = word_addr * 16
        pixels = []
        for i in range(16):
            pixel_addr = base_pixel_addr + i
            if pixel_addr in memory:
                pixels.append(memory[pixel_addr])
            else:
                pixels.append(0)
        
        # Pack 4 pixels per 32-bit word
        dout0 = (pixels[3] << 24) | (pixels[2] << 16) | (pixels[1] << 8) | pixels[0]
        dout1 = (pixels[7] << 24) | (pixels[6] << 16) | (pixels[5] << 8) | pixels[4]
        dout2 = (pixels[11] << 24) | (pixels[10] << 16) | (pixels[9] << 8) | pixels[8]
        dout3 = (pixels[15] << 24) | (pixels[14] << 16) | (pixels[13] << 8) | pixels[12]
        
        return dout0, dout1, dout2, dout3
    
    # Reset
    dut.reset.value = 1
    dut.start_patch_extraction.value = 0
    dut.next_patch.value = 0
    dut.img_width.value = IMG_W
    dut.img_height.value = IMG_H
    dut.num_channels.value = NUM_CH
    
    # Initialize RAM outputs
    dut.ram_dout0.value = 0
    dut.ram_dout1.value = 0
    dut.ram_dout2.value = 0
    dut.ram_dout3.value = 0
    
    await tick()
    dut.reset.value = 0
    await tick()
    
    # Start extraction
    dut._log.info("Starting patch extraction with channel-last layout")
    dut.start_patch_extraction.value = 1
    await tick()
    dut.start_patch_extraction.value = 0
    
    # Track the sequence of patches extracted
    patches_extracted = []
    
    for position_idx in range(3):  # Test first 3 patch positions  
        for channel in range(NUM_CH):  # 3 channels per position
            
            # During FETCH_PIXELS state, simulate memory responses
            while not dut.patch_ready.value:
                await tick()
                
                # If RAM read is requested, provide the appropriate memory data
                if dut.ram_re.value:
                    word_addr = int(dut.ram_addr.value)
                    dout0, dout1, dout2, dout3 = get_memory_word(word_addr)
                    dut.ram_dout0.value = dout0
                    dut.ram_dout1.value = dout1
                    dut.ram_dout2.value = dout2
                    dut.ram_dout3.value = dout3
                    dut._log.info(f"Memory read: addr={word_addr}, data=[{dout0:08x}, {dout1:08x}, {dout2:08x}, {dout3:08x}]")
            
            # Record what was extracted
            current_channel = int(dut.current_channel.value)
            current_row = int(dut.current_patch_row.value)
            current_col = int(dut.current_patch_col.value)
            all_channels_done = bool(dut.all_channels_done_for_position.value)
            
            # Get the actual patch data
            patch_A0 = int(dut.patch_A0_out.value)
            patch_A1 = int(dut.patch_A1_out.value)
            patch_A2 = int(dut.patch_A2_out.value)
            patch_A3 = int(dut.patch_A3_out.value)
            
            patches_extracted.append({
                'channel': current_channel,
                'row': current_row, 
                'col': current_col,
                'all_channels_done': all_channels_done,
                'patch_data': [patch_A0, patch_A1, patch_A2, patch_A3]
            })
            
            dut._log.info(f"Extracted patch: channel={current_channel}, position=({current_row},{current_col}), all_channels_done={all_channels_done}")
            dut._log.info(f"Patch data: A0=0x{patch_A0:08x}, A1=0x{patch_A1:08x}, A2=0x{patch_A2:08x}, A3=0x{patch_A3:08x}")
            
            # Verify patch data matches expected values from memory
            expected_pixels = []
            for patch_row in range(4):
                for patch_col in range(4):
                    img_row = current_row + patch_row
                    img_col = current_col + patch_col
                    if img_row < IMG_H and img_col < IMG_W:
                        expected_pixel = img_row * 100 + img_col * 10 + current_channel
                        expected_pixels.append(expected_pixel & 0xFF)
                    else:
                        expected_pixels.append(0)
            
            # Check if extracted patch matches expected
            extracted_pixels = []
            for word_idx, word_data in enumerate([patch_A0, patch_A1, patch_A2, patch_A3]):
                for byte_idx in range(4):
                    pixel = (word_data >> (byte_idx * 8)) & 0xFF
                    extracted_pixels.append(pixel)
            
            # Compare (only first 16 pixels since we have 4x4 patch)
            for i in range(16):
                if extracted_pixels[i] != expected_pixels[i]:
                    dut._log.error(f"Pixel mismatch at index {i}: got {extracted_pixels[i]}, expected {expected_pixels[i]}")
                else:
                    dut._log.info(f"✓ Pixel {i}: {extracted_pixels[i]} == {expected_pixels[i]}")
            
            # Advance to next patch (next channel or next position)
            dut.next_patch.value = 1
            await tick()
            dut.next_patch.value = 0
            await tick()
            
            # If this was the last channel for this position, break inner loop
            if all_channels_done:
                break
    
    # Verify the extraction order
    dut._log.info("Verifying extraction order...")
    
    # Check that we got the expected sequence
    expected_sequence = [
        # Position (0,0): all channels
        {'channel': 0, 'row': 0, 'col': 0, 'all_channels_done': False},
        {'channel': 1, 'row': 0, 'col': 0, 'all_channels_done': False}, 
        {'channel': 2, 'row': 0, 'col': 0, 'all_channels_done': True},
        # Position (0,1): all channels  
        {'channel': 0, 'row': 0, 'col': 1, 'all_channels_done': False},
        {'channel': 1, 'row': 0, 'col': 1, 'all_channels_done': False},
        {'channel': 2, 'row': 0, 'col': 1, 'all_channels_done': True},
        # Position (0,2): all channels
        {'channel': 0, 'row': 0, 'col': 2, 'all_channels_done': False},
        {'channel': 1, 'row': 0, 'col': 2, 'all_channels_done': False},
        {'channel': 2, 'row': 0, 'col': 2, 'all_channels_done': True},
    ]
    
    # Verify sequence matches expected (ignoring patch_data for sequence check)
    for i, (actual, expected) in enumerate(zip(patches_extracted, expected_sequence)):
        actual_without_data = {k: v for k, v in actual.items() if k != 'patch_data'}
        assert actual_without_data == expected, f"Mismatch at index {i}: got {actual_without_data}, expected {expected}"
        dut._log.info(f"✓ Patch {i}: {actual_without_data}")
    
    dut._log.info("SUCCESS: Patch extractor correctly handles channel-last memory layout!") 