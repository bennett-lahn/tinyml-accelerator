import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
from cocotb.binary import BinaryValue
import numpy as np

@cocotb.test()
async def test_full_tensor_traverse(dut):
    """Complete tensor traversal with full memory trace"""
    
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
    
    cocotb.log.info("=== FULL TENSOR TRAVERSAL TRACE ===")
    cocotb.log.info(f"Image dimensions: {img_width}x{img_height}x{num_channels}")
    cocotb.log.info(f"Total spatial positions: {img_width * img_height}")
    cocotb.log.info(f"Total channel groups: {(num_channels + 3) // 4}")
    cocotb.log.info(f"Expected spatial blocks: {(img_width // 4) * (img_height // 4)}")
    cocotb.log.info("")
    
    # Start extraction
    dut.start_extraction.value = 1
    await RisingEdge(dut.clk)
    dut.start_extraction.value = 0
    
    spatial_block_count = 0
    total_channel_groups_processed = 0
    all_ram_accesses = []
    
    # Track all spatial blocks
    for block_row in range(0, img_height, 4):  # 4x4 blocks
        for block_col in range(0, img_width, 4):
            spatial_block_count += 1
            
            cocotb.log.info(f"=== SPATIAL BLOCK {spatial_block_count}: ({block_row},{block_col}) ===")
            
            # Process both channel groups for this spatial block
            for channel_group in range(2):  # 0 and 1 for 8 channels
                total_channel_groups_processed += 1
                
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
                        all_ram_accesses.append({
                            'spatial_block': spatial_block_count,
                            'channel_group': channel_group,
                            'address': addr,
                            'cycle': loading_cycles
                        })
                        
                        # Log detailed RAM access
                        ram_data = {
                            'dout0': int(dut.ram_dout0.value),
                            'dout1': int(dut.ram_dout1.value), 
                            'dout2': int(dut.ram_dout2.value),
                            'dout3': int(dut.ram_dout3.value)
                        }
                        
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
                cocotb.log.info(f"  RAM addresses accessed: {len(ram_addresses_this_group)} unique addresses")
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
                pixel_index = block_row * img_width + block_col
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
            
            # Move to next spatial block (except for last one)
            if not (block_row == img_height - 4 and block_col == img_width - 4):
                dut.next_spatial_block.value = 1
                await RisingEdge(dut.clk)
                dut.next_spatial_block.value = 0
                
                # Wait for transition
                await RisingEdge(dut.clk)
                await RisingEdge(dut.clk)
            
            cocotb.log.info("")
    
    # Final summary
    cocotb.log.info("=== TRAVERSAL SUMMARY ===")
    cocotb.log.info(f"Total spatial blocks processed: {spatial_block_count}")
    cocotb.log.info(f"Total channel groups processed: {total_channel_groups_processed}")
    cocotb.log.info(f"Total RAM accesses: {len(all_ram_accesses)}")
    
    # Analyze RAM access patterns
    unique_addresses = set(access['address'] for access in all_ram_accesses)
    cocotb.log.info(f"Unique RAM addresses accessed: {len(unique_addresses)}")
    cocotb.log.info(f"Address range: {min(unique_addresses)} - {max(unique_addresses)}")
    
    # Group accesses by spatial block
    cocotb.log.info("")
    cocotb.log.info("=== RAM ACCESS PATTERN BY SPATIAL BLOCK ===")
    for block_num in range(1, spatial_block_count + 1):
        block_accesses = [acc for acc in all_ram_accesses if acc['spatial_block'] == block_num]
        block_addresses = [acc['address'] for acc in block_accesses]
        
        # Group by channel group
        cg0_addresses = [acc['address'] for acc in block_accesses if acc['channel_group'] == 0]
        cg1_addresses = [acc['address'] for acc in block_accesses if acc['channel_group'] == 1]
        
        cocotb.log.info(f"Block {block_num}:")
        cocotb.log.info(f"  Channel Group 0: {len(cg0_addresses)} accesses, addresses {cg0_addresses[:5]}{'...' if len(cg0_addresses) > 5 else ''}")
        cocotb.log.info(f"  Channel Group 1: {len(cg1_addresses)} accesses, addresses {cg1_addresses[:5]}{'...' if len(cg1_addresses) > 5 else ''}")
        
        # Check if both channel groups access same addresses
        if set(cg0_addresses) == set(cg1_addresses):
            cocotb.log.info(f"  ✓ Both channel groups access same addresses")
        else:
            cocotb.log.info(f"  ✗ Channel groups access different addresses")
    
    # Memory efficiency analysis
    cocotb.log.info("")
    cocotb.log.info("=== MEMORY EFFICIENCY ANALYSIS ===")
    expected_unique_addresses = img_width * img_height // 2  # Each address covers 2 pixels
    cocotb.log.info(f"Expected unique addresses for full image: {expected_unique_addresses}")
    cocotb.log.info(f"Actual unique addresses accessed: {len(unique_addresses)}")
    
    if len(unique_addresses) == expected_unique_addresses:
        cocotb.log.info("✓ Memory access pattern is optimal - no redundant reads")
    else:
        cocotb.log.info("⚠ Memory access pattern may have redundancy")
    
    # Address distribution analysis
    cocotb.log.info("")
    cocotb.log.info("=== ADDRESS DISTRIBUTION ===")
    address_counts = {}
    for access in all_ram_accesses:
        addr = access['address']
        address_counts[addr] = address_counts.get(addr, 0) + 1
    
    # Show most frequently accessed addresses
    sorted_addresses = sorted(address_counts.items(), key=lambda x: x[1], reverse=True)
    cocotb.log.info("Most frequently accessed addresses:")
    for addr, count in sorted_addresses[:10]:
        cocotb.log.info(f"  Address {addr}: accessed {count} times")
    
    cocotb.log.info("")
    cocotb.log.info("Full tensor traversal trace completed!")

@cocotb.test()
async def test_memory_reuse_analysis(dut):
    """Analyze memory reuse patterns across spatial blocks"""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    # Configure for smaller image for detailed analysis
    img_width = 8
    img_height = 8
    num_channels = 8
    
    dut.img_width.value = img_width
    dut.img_height.value = img_height
    dut.num_channels.value = num_channels
    dut.pad_top.value = 0
    dut.pad_bottom.value = 0
    dut.pad_left.value = 0
    dut.pad_right.value = 0
    
    cocotb.log.info("=== MEMORY REUSE ANALYSIS ===")
    cocotb.log.info(f"Image dimensions: {img_width}x{img_height}x{num_channels}")
    
    # Start extraction
    dut.start_extraction.value = 1
    await RisingEdge(dut.clk)
    dut.start_extraction.value = 0
    
    spatial_blocks = []
    
    # Process first few spatial blocks to analyze overlap
    for block_row in range(0, min(8, img_height), 4):
        for block_col in range(0, min(8, img_width), 4):
            
            block_info = {
                'position': (block_row, block_col),
                'addresses': {'cg0': [], 'cg1': []}
            }
            
            # Process both channel groups
            for channel_group in range(2):
                
                # Wait for loading to complete
                while True:
                    await RisingEdge(dut.clk)
                    
                    if dut.unified_buffer_inst.ram_re.value == 1:
                        addr = int(dut.unified_buffer_inst.ram_addr.value)
                        cg_key = f'cg{channel_group}'
                        block_info['addresses'][cg_key].append(addr)
                    
                    if dut.buffer_loading_complete.value == 1:
                        break
                
                # Move to next channel group
                if channel_group == 0:
                    dut.next_channel_group.value = 1
                    await RisingEdge(dut.clk)
                    dut.next_channel_group.value = 0
                    await RisingEdge(dut.clk)
            
            spatial_blocks.append(block_info)
            
            # Move to next spatial block
            dut.next_spatial_block.value = 1
            await RisingEdge(dut.clk)
            dut.next_spatial_block.value = 0
            await RisingEdge(dut.clk)
    
    # Analyze overlaps between adjacent blocks
    cocotb.log.info("")
    cocotb.log.info("=== SPATIAL BLOCK OVERLAP ANALYSIS ===")
    
    for i, block in enumerate(spatial_blocks):
        pos = block['position']
        addresses = set(block['addresses']['cg0'])  # Use channel group 0 addresses
        
        cocotb.log.info(f"Block {i+1} at {pos}:")
        cocotb.log.info(f"  Addresses: {sorted(addresses)}")
        
        # Check overlap with previous blocks
        for j, prev_block in enumerate(spatial_blocks[:i]):
            prev_addresses = set(prev_block['addresses']['cg0'])
            overlap = addresses.intersection(prev_addresses)
            
            if overlap:
                overlap_percent = len(overlap) / len(addresses) * 100
                cocotb.log.info(f"  Overlap with block {j+1}: {len(overlap)} addresses ({overlap_percent:.1f}%)")
                cocotb.log.info(f"    Shared addresses: {sorted(overlap)}")
    
    cocotb.log.info("")
    cocotb.log.info("Memory reuse analysis completed!") 