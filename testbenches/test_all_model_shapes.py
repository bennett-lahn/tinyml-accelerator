import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
from cocotb.binary import BinaryValue
import numpy as np

@cocotb.test()
async def test_all_model_input_shapes(dut):
    """Test all input shapes required by the TPU target model"""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Define all input shapes from the model
    model_input_shapes = [
        # (height, width, channels, layer_name)
        (32, 32, 1, "Layer1_Input"),    # Initial input
        (16, 16, 8, "Layer2_Input"),    # After conv1 + pool1
        (8, 8, 16, "Layer3_Input"),     # After conv2 + pool2  
        (4, 4, 32, "Layer4_Input"),     # After conv3 + pool3
    ]
    
    cocotb.log.info("=== TESTING ALL TPU MODEL INPUT SHAPES ===")
    cocotb.log.info("Testing data pipeline compatibility with all CNN layer inputs")
    cocotb.log.info("")
    
    for height, width, channels, layer_name in model_input_shapes:
        cocotb.log.info(f"=== TESTING {layer_name}: {height}x{width}x{channels} ===")
        
        # Reset
        dut.reset.value = 1
        await RisingEdge(dut.clk)
        await RisingEdge(dut.clk)
        dut.reset.value = 0
        await RisingEdge(dut.clk)
        
        # Configure for current shape
        dut.img_width.value = width
        dut.img_height.value = height
        dut.num_channels.value = channels
        dut.pad_top.value = 0
        dut.pad_bottom.value = 0
        dut.pad_left.value = 0
        dut.pad_right.value = 0
        
        # Calculate expected parameters
        channel_groups = (channels + 3) // 4  # Ceiling division
        spatial_blocks_x = (width + 3) // 4   # Ceiling division for 4x4 patches
        spatial_blocks_y = (height + 3) // 4  # Ceiling division for 4x4 patches
        total_spatial_blocks = spatial_blocks_x * spatial_blocks_y
        
        cocotb.log.info(f"  Expected channel groups: {channel_groups}")
        cocotb.log.info(f"  Expected spatial blocks: {spatial_blocks_x}x{spatial_blocks_y} = {total_spatial_blocks}")
        
        # Start extraction
        dut.start_extraction.value = 1
        await RisingEdge(dut.clk)
        dut.start_extraction.value = 0
        
        # Test first spatial block with all channel groups
        spatial_block_success = True
        
        for cg in range(channel_groups):
            cocotb.log.info(f"    Testing channel group {cg}/{channel_groups-1}")
            
            # Wait for loading to complete
            loading_cycles = 0
            while True:
                await RisingEdge(dut.clk)
                loading_cycles += 1
                
                if dut.buffer_loading_complete.value == 1:
                    break
                    
                if loading_cycles > 500:  # Increased timeout for larger shapes
                    cocotb.log.error(f"    ✗ Timeout waiting for channel group {cg} loading")
                    spatial_block_success = False
                    break
            
            if not spatial_block_success:
                break
                
            cocotb.log.info(f"    ✓ Channel group {cg} loaded in {loading_cycles} cycles")
            
            # Check that data is available
            patch_data = int(dut.patch_pe00_out.value)
            if patch_data == 0:
                cocotb.log.warning(f"    Warning: patch_pe00_out is 0 for channel group {cg}")
            
            # Move to next channel group (except for last one)
            if cg < channel_groups - 1:
                dut.next_channel_group.value = 1
                await RisingEdge(dut.clk)
                dut.next_channel_group.value = 0
                await RisingEdge(dut.clk)
                await RisingEdge(dut.clk)
        
        if spatial_block_success:
            cocotb.log.info(f"  ✓ {layer_name} ({height}x{width}x{channels}) - PASSED")
        else:
            cocotb.log.error(f"  ✗ {layer_name} ({height}x{width}x{channels}) - FAILED")
        
        cocotb.log.info("")
    
    cocotb.log.info("=== MODEL SHAPE COMPATIBILITY TEST COMPLETE ===")

@cocotb.test()
async def test_parameter_limits(dut):
    """Test the parameter limits of the unified buffer"""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    cocotb.log.info("=== TESTING PARAMETER LIMITS ===")
    
    # Test cases: (height, width, channels, description, should_pass)
    limit_test_cases = [
        # Within limits
        (32, 32, 1, "Small single channel", True),
        (32, 32, 64, "Max channels", True),
        (64, 64, 1, "Max dimensions", True),
        (64, 64, 64, "Max everything", True),
        
        # Edge cases
        (1, 1, 1, "Minimum size", True),
        (4, 4, 4, "Single spatial block", True),
        (7, 7, 8, "Single 7x7 buffer", True),
        
        # Potential issues (if any)
        (63, 63, 63, "Near max odd dimensions", True),
    ]
    
    for height, width, channels, description, should_pass in limit_test_cases:
        cocotb.log.info(f"Testing: {description} ({height}x{width}x{channels})")
        
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
        
        # Start extraction
        dut.start_extraction.value = 1
        await RisingEdge(dut.clk)
        dut.start_extraction.value = 0
        
        # Wait for first loading to complete
        loading_cycles = 0
        success = False
        
        while loading_cycles < 1000:  # Large timeout
            await RisingEdge(dut.clk)
            loading_cycles += 1
            
            if dut.buffer_loading_complete.value == 1:
                success = True
                break
        
        if success == should_pass:
            status = "✓ PASSED" if success else "✓ FAILED (as expected)"
            cocotb.log.info(f"  {status} - {description}")
        else:
            status = "✗ UNEXPECTED" 
            cocotb.log.error(f"  {status} - {description}")
        
        cocotb.log.info(f"    Loading cycles: {loading_cycles}")
        cocotb.log.info("")
    
    cocotb.log.info("=== PARAMETER LIMITS TEST COMPLETE ===")

@cocotb.test()
async def test_padding_compatibility(dut):
    """Test padding support for different input shapes"""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    cocotb.log.info("=== TESTING PADDING COMPATIBILITY ===")
    
    # Test padding scenarios that might occur in the model
    padding_test_cases = [
        # (height, width, channels, pad_top, pad_bottom, pad_left, pad_right, description)
        (32, 32, 1, 1, 1, 1, 1, "32x32 with 1-pixel padding"),
        (16, 16, 8, 2, 2, 2, 2, "16x16 with 2-pixel padding"),
        (8, 8, 16, 1, 1, 1, 1, "8x8 with 1-pixel padding"),
        (4, 4, 32, 0, 0, 0, 0, "4x4 with no padding"),
        (30, 30, 1, 1, 1, 1, 1, "Non-power-of-2 with padding"),
    ]
    
    for height, width, channels, pad_top, pad_bottom, pad_left, pad_right, description in padding_test_cases:
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
        dut.pad_top.value = pad_top
        dut.pad_bottom.value = pad_bottom
        dut.pad_left.value = pad_left
        dut.pad_right.value = pad_right
        
        # Calculate padded dimensions
        padded_width = width + pad_left + pad_right
        padded_height = height + pad_top + pad_bottom
        
        cocotb.log.info(f"  Original: {height}x{width}, Padded: {padded_height}x{padded_width}")
        
        # Start extraction
        dut.start_extraction.value = 1
        await RisingEdge(dut.clk)
        dut.start_extraction.value = 0
        
        # Wait for first loading to complete
        loading_cycles = 0
        success = False
        
        while loading_cycles < 500:
            await RisingEdge(dut.clk)
            loading_cycles += 1
            
            if dut.buffer_loading_complete.value == 1:
                success = True
                break
        
        if success:
            cocotb.log.info(f"  ✓ PASSED - {description} (loaded in {loading_cycles} cycles)")
        else:
            cocotb.log.error(f"  ✗ FAILED - {description} (timeout after {loading_cycles} cycles)")
        
        cocotb.log.info("")
    
    cocotb.log.info("=== PADDING COMPATIBILITY TEST COMPLETE ===")

@cocotb.test()
async def test_address_range_validation(dut):
    """Test that address calculations stay within valid ranges for all model shapes"""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    cocotb.log.info("=== TESTING ADDRESS RANGE VALIDATION ===")
    
    model_shapes = [
        (32, 32, 1),
        (16, 16, 8), 
        (8, 8, 16),
        (4, 4, 32)
    ]
    
    for height, width, channels in model_shapes:
        cocotb.log.info(f"Testing address ranges for {height}x{width}x{channels}")
        
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
        
        # Calculate expected address range
        max_expected_addr = height * width - 1
        
        cocotb.log.info(f"  Expected address range: 0 to {max_expected_addr}")
        
        # Start extraction
        dut.start_extraction.value = 1
        await RisingEdge(dut.clk)
        dut.start_extraction.value = 0
        
        # Monitor addresses during loading
        observed_addresses = set()
        loading_cycles = 0
        
        while loading_cycles < 200:
            await RisingEdge(dut.clk)
            loading_cycles += 1
            
            # Check RAM access
            if dut.unified_buffer_inst.ram_re.value == 1:
                addr = int(dut.unified_buffer_inst.ram_addr.value)
                observed_addresses.add(addr)
                
                # Validate address is in expected range
                if addr > max_expected_addr:
                    cocotb.log.error(f"  ✗ Address {addr} exceeds maximum {max_expected_addr}")
            
            if dut.buffer_loading_complete.value == 1:
                break
        
        if observed_addresses:
            min_addr = min(observed_addresses)
            max_addr = max(observed_addresses)
            cocotb.log.info(f"  Observed address range: {min_addr} to {max_addr}")
            
            if max_addr <= max_expected_addr:
                cocotb.log.info(f"  ✓ All addresses within valid range")
            else:
                cocotb.log.error(f"  ✗ Some addresses exceed valid range")
        else:
            cocotb.log.warning(f"  Warning: No RAM accesses observed")
        
        cocotb.log.info("")
    
    cocotb.log.info("=== ADDRESS RANGE VALIDATION COMPLETE ===") 