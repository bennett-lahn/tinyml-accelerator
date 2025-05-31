import cocotb
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge, Timer
from cocotb.binary import BinaryValue
import numpy as np

@cocotb.test()
async def test_debug_loading_completion(dut):
    """Debug test to understand loading completion behavior"""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    # Configure for 32x32x1 image (single channel)
    dut.img_width.value = 32
    dut.img_height.value = 32
    dut.num_channels.value = 1
    dut.pad_top.value = 0
    dut.pad_bottom.value = 0
    dut.pad_left.value = 0
    dut.pad_right.value = 0
    
    # Initialize control signals
    dut.start_extraction.value = 0
    dut.next_channel_group.value = 0
    dut.next_spatial_block.value = 0
    dut.start_formatting.value = 0
    dut.tensor_ram_we.value = 0
    
    await RisingEdge(dut.clk)
    
    # Start extraction
    cocotb.log.info("Starting extraction...")
    
    dut.start_extraction.value = 1
    await RisingEdge(dut.clk)
    dut.start_extraction.value = 0
    
    # Monitor loading progress
    cycle_count = 0
    max_cycles = 200
    
    while cycle_count < max_cycles:
        await RisingEdge(dut.clk)
        cycle_count += 1
        
        # Access unified buffer internal signals
        ub = dut.unified_buffer_inst
        
        state = int(ub.state.value)
        load_row = int(ub.load_row.value)
        load_col = int(ub.load_col.value)
        loading_cycle_complete = int(ub.loading_cycle_complete.value)
        all_data_loaded = int(ub.all_data_loaded.value)
        completion_delay_counter = int(ub.completion_delay_counter.value)
        buffer_loading_complete = int(dut.buffer_loading_complete.value)
        block_ready = int(dut.block_ready.value)
        
        # Log key cycles
        if cycle_count <= 10 or loading_cycle_complete or all_data_loaded or buffer_loading_complete or block_ready:
            cocotb.log.info(f"Cycle {cycle_count:3d}: state={state}, pos=({load_row},{load_col}), "
                          f"cycle_complete={loading_cycle_complete}, all_loaded={all_data_loaded}, "
                          f"delay_counter={completion_delay_counter}, buffer_complete={buffer_loading_complete}, "
                          f"block_ready={block_ready}")
        
        if buffer_loading_complete and block_ready:
            cocotb.log.info(f"Loading completed at cycle {cycle_count}")
            break
    
    cocotb.log.info("Debug loading test completed!")

@cocotb.test()
async def test_debug_spatial_advancement(dut):
    """Debug test to understand spatial block advancement"""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    # Configure for 32x32x1 image (single channel)
    dut.img_width.value = 32
    dut.img_height.value = 32
    dut.num_channels.value = 1
    dut.pad_top.value = 0
    dut.pad_bottom.value = 0
    dut.pad_left.value = 0
    dut.pad_right.value = 0
    
    # Initialize control signals
    dut.start_extraction.value = 0
    dut.next_channel_group.value = 0
    dut.next_spatial_block.value = 0
    dut.start_formatting.value = 0
    dut.tensor_ram_we.value = 0
    
    await RisingEdge(dut.clk)
    
    # Start extraction
    cocotb.log.info("Starting extraction...")
    
    dut.start_extraction.value = 1
    await RisingEdge(dut.clk)
    dut.start_extraction.value = 0
    
    # Wait for first block to load
    cycle_count = 0
    while not dut.buffer_loading_complete.value and cycle_count < 200:
        await RisingEdge(dut.clk)
        cycle_count += 1
    
    # Wait one more cycle for state transition
    await RisingEdge(dut.clk)
    
    # Access unified buffer internal signals
    ub = dut.unified_buffer_inst
    
    # Log initial spatial position
    block_start_row = int(ub.block_start_row.value.signed_integer) if hasattr(ub.block_start_row.value, 'signed_integer') else int(ub.block_start_row.value)
    block_start_col = int(ub.block_start_col.value.signed_integer) if hasattr(ub.block_start_col.value, 'signed_integer') else int(ub.block_start_col.value)
    channel_group = int(ub.channel_group.value)
    total_channel_groups = int(ub.total_channel_groups.value)
    state = int(ub.state.value)
    
    cocotb.log.info(f"Initial state: block_start=({block_start_row},{block_start_col}), "
                  f"channel_group={channel_group}/{total_channel_groups}, state={state}")
    
    # Get first block data
    first_patch_00 = int(dut.patch_pe00_out.value)
    cocotb.log.info(f"First block patch_pe00_out: 0x{first_patch_00:08x}")
    
    # Trigger next channel group
    cocotb.log.info("Triggering next_channel_group...")
    dut.next_channel_group.value = 1
    await RisingEdge(dut.clk)
    dut.next_channel_group.value = 0
    
    # Check state after next_channel_group
    await RisingEdge(dut.clk)
    
    state = int(ub.state.value)
    all_channels_done = int(dut.all_channels_done.value)
    
    cocotb.log.info(f"After next_channel_group: state={state}, all_channels_done={all_channels_done}")
    
    if all_channels_done:
        cocotb.log.info("All channels done - triggering next_spatial_block...")
        
        # Log current spatial position before advancement
        block_start_row_before = int(ub.block_start_row.value.signed_integer) if hasattr(ub.block_start_row.value, 'signed_integer') else int(ub.block_start_row.value)
        block_start_col_before = int(ub.block_start_col.value.signed_integer) if hasattr(ub.block_start_col.value, 'signed_integer') else int(ub.block_start_col.value)
        
        cocotb.log.info(f"Before next_spatial_block: block_start=({block_start_row_before},{block_start_col_before})")
        
        # Trigger next spatial block
        dut.next_spatial_block.value = 1
        await RisingEdge(dut.clk)
        dut.next_spatial_block.value = 0
        
        # Check spatial position after advancement
        await RisingEdge(dut.clk)
        
        block_start_row_after = int(ub.block_start_row.value.signed_integer) if hasattr(ub.block_start_row.value, 'signed_integer') else int(ub.block_start_row.value)
        block_start_col_after = int(ub.block_start_col.value.signed_integer) if hasattr(ub.block_start_col.value, 'signed_integer') else int(ub.block_start_col.value)
        state_after = int(ub.state.value)
        
        cocotb.log.info(f"After next_spatial_block: block_start=({block_start_row_after},{block_start_col_after}), state={state_after}")
        
        # Check if spatial position actually changed
        if block_start_row_after != block_start_row_before or block_start_col_after != block_start_col_before:
            cocotb.log.info("✓ Spatial position advanced correctly!")
        else:
            cocotb.log.error("✗ Spatial position did NOT advance!")
        
        # Wait for next block to load
        cycle_count = 0
        while not dut.buffer_loading_complete.value and cycle_count < 200:
            await RisingEdge(dut.clk)
            cycle_count += 1
        
        # Wait one more cycle for state transition
        await RisingEdge(dut.clk)
        
        # Get next block data
        next_patch_00 = int(dut.patch_pe00_out.value)
        cocotb.log.info(f"Next block patch_pe00_out: 0x{next_patch_00:08x}")
        
        # Compare data
        if next_patch_00 != first_patch_00:
            cocotb.log.info("✓ Data changed - next spatial block loaded different data!")
        else:
            cocotb.log.info("✗ Data unchanged - next spatial block loaded same data")
    
    cocotb.log.info("Spatial advancement debug test completed!")

@cocotb.test()
async def test_simple_loading_counter(dut):
    """Test just the loading counter behavior"""
    
    # Start clock
    clock = Clock(dut.clk, 10, units="ns")
    cocotb.start_soon(clock.start())
    
    # Reset
    dut.reset.value = 1
    await RisingEdge(dut.clk)
    await RisingEdge(dut.clk)
    dut.reset.value = 0
    await RisingEdge(dut.clk)
    
    # Configure for 32x32x1 image
    dut.img_width.value = 32
    dut.img_height.value = 32
    dut.num_channels.value = 1
    dut.pad_top.value = 0
    dut.pad_bottom.value = 0
    dut.pad_left.value = 0
    dut.pad_right.value = 0
    
    # Initialize control signals
    dut.start_extraction.value = 0
    dut.next_channel_group.value = 0
    dut.next_spatial_block.value = 0
    dut.start_formatting.value = 0
    dut.tensor_ram_we.value = 0
    
    await RisingEdge(dut.clk)
    
    # Start extraction
    dut.start_extraction.value = 1
    await RisingEdge(dut.clk)
    dut.start_extraction.value = 0
    
    # Count how many cycles it takes to go through all 49 positions
    positions_seen = set()
    cycle_count = 0
    max_cycles = 100
    
    while cycle_count < max_cycles:
        await RisingEdge(dut.clk)
        cycle_count += 1
        
        try:
            state = int(dut.unified_buffer_inst.state.value)
            if state == 1:  # LOADING_BLOCK
                load_row = int(dut.unified_buffer_inst.load_row.value)
                load_col = int(dut.unified_buffer_inst.load_col.value)
                position = (load_row, load_col)
                positions_seen.add(position)
                
                if len(positions_seen) % 10 == 0:
                    cocotb.log.info(f"Cycle {cycle_count}: Seen {len(positions_seen)} positions, current: {position}")
                
                # Check if we've seen all 49 positions
                if len(positions_seen) == 49:
                    cocotb.log.info(f"All 49 positions seen after {cycle_count} cycles")
                    break
            elif state == 2:  # BLOCK_READY
                cocotb.log.info(f"State changed to BLOCK_READY after {cycle_count} cycles")
                break
                
        except Exception as e:
            cocotb.log.warning(f"Could not access internal signals: {e}")
            break
    
    cocotb.log.info(f"Total positions seen: {len(positions_seen)}")
    cocotb.log.info(f"Positions: {sorted(list(positions_seen))}")
    
    # Verify we saw all expected positions
    expected_positions = {(r, c) for r in range(7) for c in range(7)}
    missing_positions = expected_positions - positions_seen
    if missing_positions:
        cocotb.log.error(f"Missing positions: {missing_positions}")
    
    extra_positions = positions_seen - expected_positions
    if extra_positions:
        cocotb.log.error(f"Extra positions: {extra_positions}")
    
    assert len(positions_seen) == 49, f"Expected 49 positions, got {len(positions_seen)}"
    assert missing_positions == set(), f"Missing positions: {missing_positions}" 