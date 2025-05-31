import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
from cocotb.binary import BinaryValue
import numpy as np

@cocotb.test()
async def test_simple_memory_trace(dut):
    """Simple memory trace for first spatial block only"""
    
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
    img_width = 16
    img_height = 16
    num_channels = 8
    
    dut.img_width.value = img_width
    dut.img_height.value = img_height
    dut.num_channels.value = num_channels
    dut.pad_top.value = 0
    dut.pad_bottom.value = 0
    dut.pad_left.value = 0
    dut.pad_right.value = 0
    
    cocotb.log.info("=== SIMPLE MEMORY TRACE - FIRST SPATIAL BLOCK ===")
    cocotb.log.info(f"Image dimensions: {img_width}x{img_height}x{num_channels}")
    cocotb.log.info("")
    
    # Start extraction
    dut.start_extraction.value = 1
    await RisingEdge(dut.clk)
    dut.start_extraction.value = 0
    
    # Process first spatial block (0,0) with both channel groups
    cocotb.log.info("=== SPATIAL BLOCK 1: (0,0) ===")
    
    all_accesses = []
    
    # Process both channel groups for this spatial block
    for channel_group in range(2):  # 0 and 1 for 8 channels
        
        cocotb.log.info(f"--- Channel Group {channel_group} (channels {channel_group*4}-{channel_group*4+3}) ---")
        
        # Wait for loading to complete
        loading_cycles = 0
        ram_addresses_this_group = []
        
        while True:
            await RisingEdge(dut.clk)
            loading_cycles += 1
            
            # Track RAM accesses through unified buffer instance
            if dut.unified_buffer_inst.ram_re.value == 1:
                addr = int(dut.unified_buffer_inst.ram_addr.value)
                ram_addresses_this_group.append(addr)
                
                # Log detailed RAM access
                ram_data = {
                    'dout0': int(dut.ram_dout0.value),
                    'dout1': int(dut.ram_dout1.value), 
                    'dout2': int(dut.ram_dout2.value),
                    'dout3': int(dut.ram_dout3.value)
                }
                
                all_accesses.append({
                    'channel_group': channel_group,
                    'cycle': loading_cycles,
                    'address': addr,
                    'data': ram_data
                })
                
                cocotb.log.info(f"  Cycle {loading_cycles:3d}: RAM[{addr:3d}] = "
                              f"dout3:0x{ram_data['dout3']:08x} "
                              f"dout2:0x{ram_data['dout2']:08x} "
                              f"dout1:0x{ram_data['dout1']:08x} "
                              f"dout0:0x{ram_data['dout0']:08x}")
            
            # Check if loading is complete
            if dut.buffer_loading_complete.value == 1:
                break
                
            # Safety timeout
            if loading_cycles > 200:
                cocotb.log.error(f"Timeout waiting for channel group {channel_group} loading")
                break
        
        cocotb.log.info(f"  Loading completed in {loading_cycles} cycles")
        cocotb.log.info(f"  RAM addresses accessed: {len(ram_addresses_this_group)}")
        cocotb.log.info(f"  Address range: {min(ram_addresses_this_group) if ram_addresses_this_group else 'N/A'} - {max(ram_addresses_this_group) if ram_addresses_this_group else 'N/A'}")
        
        # Check extracted data at position (0,0)
        patch_data = int(dut.patch_pe00_out.value)
        channels = [
            (patch_data >> 0) & 0xFF,
            (patch_data >> 8) & 0xFF,
            (patch_data >> 16) & 0xFF,
            (patch_data >> 24) & 0xFF
        ]
        cocotb.log.info(f"  Extracted channels at (0,0): {channels}")
        
        # Calculate expected channels for this position
        pixel_index = 0  # First spatial block at (0,0)
        expected_channels = [
            (pixel_index * 8 + channel_group * 4 + i) % 256 
            for i in range(4)
        ]
        cocotb.log.info(f"  Expected channels at (0,0): {expected_channels}")
        
        if channels == expected_channels:
            cocotb.log.info(f"  ✓ Channel group {channel_group} data correct!")
        else:
            cocotb.log.error(f"  ✗ Channel group {channel_group} data mismatch!")
        
        # Move to next channel group (except for last one)
        if channel_group == 0:
            dut.next_channel_group.value = 1
            await RisingEdge(dut.clk)
            dut.next_channel_group.value = 0
            
            # Wait for transition
            await RisingEdge(dut.clk)
            await RisingEdge(dut.clk)
        
        cocotb.log.info("")
    
    # Analysis
    cocotb.log.info("=== MEMORY ACCESS ANALYSIS ===")
    
    # Group by channel group
    cg0_accesses = [acc for acc in all_accesses if acc['channel_group'] == 0]
    cg1_accesses = [acc for acc in all_accesses if acc['channel_group'] == 1]
    
    cg0_addresses = [acc['address'] for acc in cg0_accesses]
    cg1_addresses = [acc['address'] for acc in cg1_accesses]
    
    cocotb.log.info(f"Channel Group 0: {len(cg0_addresses)} accesses")
    cocotb.log.info(f"  Addresses: {cg0_addresses}")
    cocotb.log.info(f"Channel Group 1: {len(cg1_addresses)} accesses")
    cocotb.log.info(f"  Addresses: {cg1_addresses}")
    
    # Check if both channel groups access same addresses
    if set(cg0_addresses) == set(cg1_addresses):
        cocotb.log.info("✓ Both channel groups access same addresses (correct)")
    else:
        cocotb.log.info("✗ Channel groups access different addresses")
        cocotb.log.info(f"  CG0 only: {set(cg0_addresses) - set(cg1_addresses)}")
        cocotb.log.info(f"  CG1 only: {set(cg1_addresses) - set(cg0_addresses)}")
    
    # Show data interpretation for first few addresses
    cocotb.log.info("")
    cocotb.log.info("=== DATA INTERPRETATION ===")
    
    unique_addresses = sorted(set(cg0_addresses))
    for addr in unique_addresses[:5]:  # Show first 5 addresses
        # Find accesses for this address
        cg0_access = next((acc for acc in cg0_accesses if acc['address'] == addr), None)
        cg1_access = next((acc for acc in cg1_accesses if acc['address'] == addr), None)
        
        if cg0_access:
            data = cg0_access['data']
            cocotb.log.info(f"Address {addr}:")
            cocotb.log.info(f"  Raw data: dout3=0x{data['dout3']:08x} dout2=0x{data['dout2']:08x} dout1=0x{data['dout1']:08x} dout0=0x{data['dout0']:08x}")
            
            # Interpret as channels
            # Channel group 0 uses dout3 (bytes 0-3)
            cg0_channels = [
                (data['dout3'] >> 24) & 0xFF,  # byte 3 -> channel 0
                (data['dout3'] >> 16) & 0xFF,  # byte 2 -> channel 1
                (data['dout3'] >> 8) & 0xFF,   # byte 1 -> channel 2
                (data['dout3'] >> 0) & 0xFF    # byte 0 -> channel 3
            ]
            
            # Channel group 1 uses dout2 (bytes 4-7)
            cg1_channels = [
                (data['dout2'] >> 24) & 0xFF,  # byte 7 -> channel 4
                (data['dout2'] >> 16) & 0xFF,  # byte 6 -> channel 5
                (data['dout2'] >> 8) & 0xFF,   # byte 5 -> channel 6
                (data['dout2'] >> 0) & 0xFF    # byte 4 -> channel 7
            ]
            
            cocotb.log.info(f"  CG0 channels (0-3): {cg0_channels}")
            cocotb.log.info(f"  CG1 channels (4-7): {cg1_channels}")
    
    cocotb.log.info("")
    cocotb.log.info("Simple memory trace completed!") 