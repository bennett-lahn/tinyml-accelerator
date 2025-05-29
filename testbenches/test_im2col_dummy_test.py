import cocotb, logging
cocotb.log.setLevel(logging.DEBUG)
from cocotb.triggers import RisingEdge
from cocotb.clock import Clock
import random


@cocotb.test()
async def test_im2col_dummy_test(dut):
    CLOCK_PERIOD = 2
    dut._log.setLevel(logging.DEBUG)
    # Start clock
    cocotb.start_soon(Clock(dut.clk, CLOCK_PERIOD, units="ns").start())

    async def tick():
        await RisingEdge(dut.clk)

    async def reset_dut():
        dut.reset.value = 1
        await tick()
        dut.reset.value = 0
        await tick()


    async def set_h_w_c_in(h, w, c):
        dut.image_height.value = h
        dut.image_width.value = w
        dut.channel_count.value = c
        await tick()
    
    async def get_next_patch():   
        dut.get_next_patch.value = 1
        await tick()
        dut.get_next_patch.value = 0
        await tick()
    
    async def set_channel_select(c):
        dut.channel_select.value = c
        await tick()
    
    

    async def run_test():

        await tick()
        await tick()

        dut._log.info("resetting dut")
        await reset_dut()

        

        dut._log.info("setting h, w, c")
        await set_h_w_c_in(32, 32, 1)

        dut._log.info("starting im2col")

        for i in range(1000):
            await tick()

        # await get_next_patch()
        # dut._log.info("getting next patch")
        # while not dut.done_patch.value:
        #     await tick()
        #     # dut._log.info("done_patch: %s", dut.done_patch.value)
        #     # dut._log.info("patch_ready: %s", dut.im2col_inst.patch_ready.value)
        #     # dut._log.info("state: %s", dut.im2col_inst.state.value)
      

        # dut._log.info("reading fifo")
        # await get_next_patch()

        # while not dut.done_patch.value:
        #     await tick()
        #     # dut._log.info("done_patch: %s", dut.done_patch.value)
        #     # dut._log.info("patch_ready: %s", dut.im2col_inst.patch_ready.value)
        #     # dut._log.info("state: %s", dut.im2col_inst.state.value)



        # for i in range(841):
        #     await tick()

        dut._log.info("resetting dut")
        await reset_dut()

    await run_test()

    

    
    
    
    # Test with random input values