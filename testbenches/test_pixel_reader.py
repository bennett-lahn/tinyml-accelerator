import cocotb
import logging
import random
from cocotb.clock import Clock
from cocotb.triggers import Timer, RisingEdge

@cocotb.test()
async def test_pixel_reader(dut):
    dut._log.setLevel(logging.INFO)
    cocotb.start_soon(Clock(dut.clk, 2, units="ns").start())

    async def tick():
        await RisingEdge(dut.clk)  
    
    async def monitor_inputs_and_outputs(dut):
        cycle = 0
        while True:
            await tick()
            log_lines = []
            log_lines.append(f"MON clk={cycle:2d}: pixel_ptr={dut.pixel_ptr.value} valid_out={dut.valid_out.value}")
            for line in log_lines:
                dut._log.info(line)
            cycle += 1
    
    async def apply_inputs_drive_only(do_reset, start):
        dut.reset.value = 1 if do_reset else 0  
        dut.start.value = start
    
    async def apply_inputs_and_tick(do_reset, start, cycles=1):
        await apply_inputs_drive_only(do_reset, start)
        for _ in range(cycles):
            await tick()
    
    async def run_test_case(start):
        # 1) Reset
        await apply_inputs_drive_only(True, 0)
        await tick()
        await apply_inputs_drive_only(False, 0)
        await tick()
        
        # 2) Apply inputs

        await apply_inputs_and_tick(False, start)

        # Wait for all pixels to be read
        while(not dut.pixel_ptr.value == 96*96-1):
            await tick()

    dut._log.info("Running test on pixel reader")
    await run_test_case(1)
    dut._log.info("Test complete")
        
        

        
    