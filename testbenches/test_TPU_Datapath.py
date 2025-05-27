import cocotb, logging
cocotb.log.setLevel(logging.DEBUG)
from cocotb.triggers import RisingEdge
from cocotb.clock import Clock
import random


@cocotb.test()
async def test_TPU_Datapath(dut):
    CLOCK_PERIOD = 2
    dut._log.setLevel(logging.DEBUG)
    # Start clock
    cocotb.start_soon(Clock(dut.clk, CLOCK_PERIOD, units="ns").start())


    async def tick():
        """Advance one clock cycle"""
        await RisingEdge(dut.clk)

    async def reset_dut():
        """Reset the DUT"""
        dut.reset.value = 1
        await tick()
        dut.reset.value = 0
        await tick()
    
    async def apply_inputs(reset, read_weights, read_inputs, start, done):
        """Apply inputs to the DUT"""
        # dut.clk.value = clk
        dut.reset.value = reset
        dut.read_weights.value = read_weights
        dut.read_inputs.value = read_inputs
        dut.start.value = start
        dut.done.value = done
        # dut.write_outputs.value = write_outputs
        await tick()


    async def run_test():
        """Run the test"""
        # Reset the DUT
        await reset_dut()

        # Test with read_weights, read_inputs, write_outputs all set to 1
        await apply_inputs(0, 1, 1, 0, 0)
        dut._log.info("Testing with all inputs set to 1")
        await tick()

        await apply_inputs(0, 1, 1, 1, 0)
        await tick()
        dut._log.info("asserted start")
        for i in range(3):
            await tick()

        await apply_inputs(0, 0, 0, 0, 1)
        for i in range(30):
            await tick()
        dut._log.info("Testing with start asserted and done set")



        # Check outputs
        # assert dut.output_valid.value == 1, "Output valid should be high"
        # dut._log.info(f"Output data: {dut.output_data.value}")
    
        # Reset the DUT again
        await reset_dut()
    await run_test()
        