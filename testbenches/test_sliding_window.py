import cocotb
import logging
import random
from cocotb.clock import Clock
from cocotb.triggers import Timer, RisingEdge



@cocotb.test()
async def test_sliding_window(dut):
    """Test the sliding window module for producint 4 lanes of 4 inputs each from memory"""
    dut._log.setLevel(logging.INFO)
    cocotb.start_soon(Clock(dut.clk, 2, units="ns").start())

    async def tick():
        await RisingEdge(dut.clk)

    async def monitor_inputs_and_outputs(dut):
        cycle = 0
        while True: # Runs indefinitely, or can be modified
            await tick()
            log_lines = []

            a0_vals = [dut.A0[i].value.signed_integer for i in range(4)]
            a1_vals = [dut.A1[i].value.signed_integer for i in range(4)]
            a2_vals = [dut.A2[i].value.signed_integer for i in range(4)]
            a3_vals = [dut.A3[i].value.signed_integer for i in range(4)]
            log_lines.append(f"MON clk={cycle:2d}: A0={a0_vals} A1={a1_vals} A2={a2_vals} A3={a3_vals}")
            log_lines.append(f"MON clk={cycle:2d}: valid_A0={dut.valid_A0.value} valid_A1={dut.valid_A1.value} valid_A2={dut.valid_A2.value} valid_A3={dut.valid_A3.value}")
            for line in log_lines:
                dut._log.info(line)
            cycle += 1
    monitor_task = cocotb.start_soon(monitor_inputs_and_outputs(dut))
    # sets inputs without advancing clock
    async def apply_inputs_drive_only(do_reset, start, addr_w, din, we = False):
        dut.reset.value = 1 if do_reset else 0
        dut.start.value = start
        dut.addr_w.value = addr_w
        dut.din.value = din
        dut.we.value = we
    
    # sets inputs and advances clock by inputted number of cycles
    async def apply_inputs_and_tick(do_reset, start, addr_w, din, we = False, cycles=1):
        await apply_inputs_drive_only(do_reset, start, addr_w, din, we)
        for _ in range(cycles):
            await tick()

        #deassert reset if needed
        if do_reset:
            dut.reset.value = 0
        

    async def run_test_case(start, addr_w, din, we = False):
        # 1) Reset
        await apply_inputs_drive_only(True, 0, 0, 0, False)
        await tick()
        await apply_inputs_drive_only(False, 0, 0, 0, False)
        await tick()

        # 2) Apply inputs
        await apply_inputs_drive_only(False, start, addr_w, din, we)
        # Wait for all pixels to be read
        while(not dut.pixel_ptr.value == 96*96-1):
            await tick()
        
    dut._log.info("Running test on window")
    await run_test_case(1, 0, 0, False)
    dut._log.info("Test complete")
        
        


