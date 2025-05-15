import cocotb
import logging
import random # For randomized test
import math   # For N_BITS calculation if needed

from cocotb.clock import Clock
from cocotb.triggers import RisingEdge

@cocotb.test()
async def test_array_output_buffer(dut):
    """Test array_output_buffer: single writes, multi-writes, simultaneous write+read, random inputs."""

    # def dump_expected_fifo(dut, expected): # Kept for potential manual debugging
    #     """
    #     Prints out the contents of the expected-FIFO for debug.
    #     dut      : the cocotb DUT handle, used to log messages
    #     expected : a Python list of tuples (value, row, col)
    #     """
    #     dut._log.info("=== Expected FIFO Dump ===")
    #     if not expected:
    #         dut._log.info("  <empty>")
    #     else:
    #         for idx, (val, row, col) in enumerate(expected):
    #             dut._log.info(f"  [{idx:2d}] value={val:8d}  row={row:3d}  col={col:3d}")
    #     dut._log.info("==========================")

    # Helper to drive up to 4 writes in parallel
    def drive_writes(valids, outputs, rows, cols):
        for i in range(4):
            dut.in_valid[i].value = 1 if valids[i] else 0
            dut.in_output[i].value = outputs[i]
            dut.in_row[i].value = rows[i]
            dut.in_col[i].value = cols[i]

    # Main checker loop - refined assertions
    async def checker_cycle():
        """
        Asserts out_consume for one cycle, ticks the clock, and checks DUT output against the 'expected' FIFO.
        """
        dut.out_consume.value = 1
        await tick()
        is_dut_output_valid = bool(dut.out_valid.value)

        if is_dut_output_valid:
            # DUT asserts out_valid. The 'expected' FIFO must not be empty.
            assert expected, "❌ FAILED: DUT asserted out_valid, but no data was expected in TB's FIFO."
            
            out_val = int(dut.out_output.value)
            out_r = int(dut.out_row.value)
            out_c = int(dut.out_col.value)
            
            exp_val, exp_r, exp_c = expected.pop(0) # Consume from expected FIFO
            
            # dut._log.info(f"Reading output {out_val} (row={out_r}, col={out_c}) for exp {exp_val} (row={exp_r}, col={exp_c})")
            assert out_val == exp_val, f"❌ FAILED: Output value mismatch: got {out_val}, exp {exp_val}"
            assert out_r   == exp_r,   f"❌ FAILED: Row mismatch: got {out_r}, exp {exp_r}"
            assert out_c   == exp_c,   f"❌ FAILED: Col mismatch: got {out_c}, exp {exp_c}"
        else:
            # DUT de-asserts out_valid. If 'expected' FIFO is not empty, it's an error.
            assert not expected, f"❌ FAILED: DUT out_valid is low, but data was expected. Expected FIFO size: {len(expected)}."
            # If execution reaches here, DUT out_valid is low and 'expected' is empty, which is fine.
            # dut._log.debug("DUT out_valid is low, and no data was expected. Correct.")
            
        dut.out_consume.value = 0 # De-assert consume for the next cycle

    async def tick():
        """Advance one clock."""
        await RisingEdge(dut.clk)

    dut._log.setLevel(logging.INFO)
    CLOCK_PERIOD = 2  # ns

    # Start the clock
    cocotb.start_soon(Clock(dut.clk, CLOCK_PERIOD, units="ns").start())

    # Reset sequence
    dut.reset.value = 1
    dut.out_consume.value = 0
    drive_writes(
        valids  = [0,0,0,0],
        outputs = [0,0,0,0],
        rows    = [0,0,0,0],
        cols    = [0,0,0,0],
    )
    await tick()
    dut.reset.value = 0
    await tick()

    # FIFO of expected entries
    expected = []

    # Test 1: single-port write
    dut._log.info("🚀 Test 1: single-port write")
    drive_writes(
        valids  = [1,0,0,0],
        outputs = [10,0,0,0],
        rows    = [2,0,0,0],
        cols    = [3,0,0,0],
    )
    expected.append((10,2,3))
    # In this first call to checker_cycle, the write of (10,2,3) and the consume happen "simultaneously"
    # from the DUT's perspective after the tick in checker_cycle.
    # Since the buffer is empty, this will test the bypass logic if out_consume is high.
    await checker_cycle() 
    dut._log.info("✅ Single-port write PASSED")

    # Test 2: four-port write
    dut._log.info("🚀 Test 2: four-port write")
    vals = [101, 202, 303, 404]
    rows = [0, 1, 2, 3]
    cols = [3, 2, 1, 0]
    drive_writes([1,1,1,1], vals, rows, cols) # Drive all 4 inputs
    await tick() # Data is written into the DUT's buffer
    # De-assert write inputs for subsequent read cycles
    drive_writes(
        valids  = [0,0,0,0],
        outputs = [0,0,0,0],
        rows    = [0,0,0,0],
        cols    = [0,0,0,0],
    )
    expected = list(zip(vals, rows, cols)) # Expected FIFO now has 4 items
    while expected:
        await checker_cycle()
    dut._log.info("✅ Four-port write PASSED")

    # Test 3: simultaneous write + read when not empty (This test writes, then reads separately)
    dut._log.info("🚀 Test 3: Fill buffer, then read (was 'simultaneous write + read when not empty')")
    vals2 = [11, 22, 33, 44]
    rows2 = [4, 5, 6, 7]
    cols2 = [7, 6, 5, 4]
    drive_writes([1,1,1,1], vals2, rows2, cols2)
    await tick()
    drive_writes([0,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0])
    expected = list(zip(vals2, rows2, cols2))
    while expected:
        await checker_cycle() # Read out items one by one
    dut._log.info("✅ Fill buffer then read PASSED")

    # Test 4: simultaneous write + read when empty (tests bypass)
    dut._log.info("🚀 Test 4: simultaneous write + read when empty (bypass test)")
    new_val, new_row, new_col = 999, 8, 9
    drive_writes([1,0,0,0], [new_val,0,0,0], [new_row,0,0,0], [new_col,0,0,0])
    expected = [(new_val, new_row, new_col)] # Expect this to be bypassed
    # checker_cycle will set out_consume=1, then tick. DUT sees write and consume.
    await checker_cycle()
    drive_writes(
        valids  = [0,0,0,0],
        outputs = [0,0,0,0],
        rows    = [0,0,0,0],
        cols    = [0,0,0,0],
    )
    dut._log.info("✅ Simultaneous write+read when empty (bypass test) PASSED")

    # Test 5: Randomized operations
    dut._log.info("🚀 Test 5: 1000 randomized operations")
    # dut._log.setLevel(logging.DEBUG) # Uncomment for more verbose random output

    num_random_ops = 1000 
    MAX_BUFFER_ENTRIES_TB = 4 
    NUM_WRITE_PORTS_TB = 4    
    
    try:
        n_bits_val = int(dut.N_BITS.value)
    except AttributeError:
        dut._log.warning("Could not read dut.N_BITS parameter, defaulting N_BITS to 4 for row/col calculation.")
        n_bits_val = 4 # Default if not readable, assuming MAX_N=16
    except ValueError:
        dut._log.warning(f"Could not convert dut.N_BITS ('{dut.N_BITS.value}') to int, defaulting N_BITS to 4.")
        n_bits_val = 4

    max_row_col_val = (1 << n_bits_val) - 1
    if max_row_col_val < 0: max_row_col_val = 0 # handles N_BITS = 0 case (e.g. if MAX_N=1)

    drive_writes([0]*NUM_WRITE_PORTS_TB, [0]*NUM_WRITE_PORTS_TB, [0]*NUM_WRITE_PORTS_TB, [0]*NUM_WRITE_PORTS_TB)
    dut.out_consume.value = 0

    for op_count in range(num_random_ops):
        dut._log.debug(f"Random op cycle {op_count+1}/{num_random_ops}, expected FIFO size: {len(expected)}")

        action = random.choices(["write", "read", "idle"], weights=[0.45, 0.45, 0.10], k=1)[0]

        if action == "write":
            available_slots = MAX_BUFFER_ENTRIES_TB - len(expected)
            if available_slots > 0:
                num_writes_this_cycle = random.randint(1, min(NUM_WRITE_PORTS_TB, available_slots))
                dut._log.debug(f"  Action: Write ({num_writes_this_cycle} item(s) into {available_slots} available slot(s))")

                current_valids = [0] * NUM_WRITE_PORTS_TB
                current_outputs = [0] * NUM_WRITE_PORTS_TB
                current_rows = [0] * NUM_WRITE_PORTS_TB
                current_cols = [0] * NUM_WRITE_PORTS_TB
                
                temp_expected_for_this_write_cycle = []
                ports_to_activate = random.sample(range(NUM_WRITE_PORTS_TB), num_writes_this_cycle)
                
                for port_idx in ports_to_activate:
                    current_valids[port_idx] = 1
                    val = random.randint(0, 10000000)
                    row = random.randint(0, max_row_col_val)
                    col = random.randint(0, max_row_col_val)
                    current_outputs[port_idx] = val
                    current_rows[port_idx] = row
                    current_cols[port_idx] = col
                    temp_expected_for_this_write_cycle.append({'val': val, 'row': row, 'col': col, 'port': port_idx})
                
                temp_expected_for_this_write_cycle.sort(key=lambda x: x['port'])
                for item in temp_expected_for_this_write_cycle:
                    expected.append((item['val'], item['row'], item['col']))

                drive_writes(current_valids, current_outputs, current_rows, current_cols)
                await tick()
                drive_writes([0]*NUM_WRITE_PORTS_TB, [0]*NUM_WRITE_PORTS_TB, [0]*NUM_WRITE_PORTS_TB, [0]*NUM_WRITE_PORTS_TB)
            else:
                dut._log.debug("  Action: Write (attempted, but buffer full). Idling.")
                drive_writes([0]*NUM_WRITE_PORTS_TB, [0]*NUM_WRITE_PORTS_TB, [0]*NUM_WRITE_PORTS_TB, [0]*NUM_WRITE_PORTS_TB)
                dut.out_consume.value = 0 # Ensure consume is off
                await tick()

        elif action == "read":
            if len(expected) > 0:
                dut._log.debug(f"  Action: Read (expected items: {len(expected)})")
                # Ensure writes are off during a dedicated read cycle
                drive_writes([0]*NUM_WRITE_PORTS_TB, [0]*NUM_WRITE_PORTS_TB, [0]*NUM_WRITE_PORTS_TB, [0]*NUM_WRITE_PORTS_TB)
                await checker_cycle() 
            else:
                dut._log.debug("  Action: Read (attempted, but expected FIFO empty). Idling.")
                drive_writes([0]*NUM_WRITE_PORTS_TB, [0]*NUM_WRITE_PORTS_TB, [0]*NUM_WRITE_PORTS_TB, [0]*NUM_WRITE_PORTS_TB)
                dut.out_consume.value = 0
                await tick()
        
        elif action == "idle":
            dut._log.debug("  Action: Idle")
            drive_writes([0]*NUM_WRITE_PORTS_TB, [0]*NUM_WRITE_PORTS_TB, [0]*NUM_WRITE_PORTS_TB, [0]*NUM_WRITE_PORTS_TB)
            dut.out_consume.value = 0
            await tick()

    dut._log.info("🚀 Test 5: Reading out any remaining items after random operations")
    while len(expected) > 0:
        dut._log.debug(f"  Clearing expected FIFO, remaining: {len(expected)}")
        # Ensure writes are off during these final read cycles
        drive_writes([0]*NUM_WRITE_PORTS_TB, [0]*NUM_WRITE_PORTS_TB, [0]*NUM_WRITE_PORTS_TB, [0]*NUM_WRITE_PORTS_TB)
        await checker_cycle()

    dut._log.info("✅ Randomized operations test PASSED")
    dut._log.setLevel(logging.INFO) # Reset logging level

    dut._log.info("✅ array_output_buffer test completed.")
