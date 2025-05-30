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
    
    async def apply_inputs(reset, read_weights, read_inputs, read_bias, load_bias, incr_weight_ptr, incr_bias_ptr, incr_input_ptr, reset_sta, start, done):
        """Apply inputs to the DUT"""
        # dut.clk.value = clk
        dut.reset.value = reset
        dut.read_weights.value = read_weights
        dut.read_inputs.value = read_inputs
        dut.read_bias.value = read_bias
        dut.load_bias.value = load_bias
        dut.incr_weight_ptr.value = incr_weight_ptr
        dut.incr_bias_ptr.value = incr_bias_ptr
        dut.incr_input_ptr.value = incr_input_ptr
        dut.reset_sta.value = reset_sta
        dut.start.value = start
        dut.done.value = done
        # dut.write_outputs.value = write_outputs
        await tick()

    async def assert_bias():
        dut._log.info("pulsing bias")
        dut.load_bias.value = 1
        await tick()
        dut.load_bias.value = 0
        await tick()

    async def assert_incr_bias_ptr():
        dut._log.info("pulsing incr_bias_ptr")
        dut.incr_bias_ptr.value = 1
        await tick()
        dut.incr_bias_ptr.value = 0
        await tick()

    async def assert_incr_input_ptr():
        dut._log.info("pulsing incr_input_ptr")
        dut.incr_input_ptr.value = 1
        await tick()
        dut.incr_input_ptr.value = 0
        await tick()

    async def assert_incr_weight_ptr():
        dut._log.info("pulsing incr_weight_ptr")
        dut.incr_weight_ptr.value = 1
        await tick()
        dut.incr_weight_ptr.value = 0
        await tick()


    async def assert_reset_sta():
        dut._log.info("pulsing reset_sta")
        dut.reset_sta.value = 1
        await tick()
        dut.reset_sta.value = 0
        await tick()

    async def assert_done():
        dut._log.info("pulsing done")
        dut.done.value = 1
        await tick()
        dut.done.value = 0
        await tick()
    
    async def assert_start():
        dut._log.info("pulsing start")
        dut.start.value = 1
        await tick()
        dut.start.value = 0
        await tick()

    async def deassert_done():
        dut._log.info("deasserting done")
        dut.done.value = 0
        await tick()
    
    async def assert_reset_ptr_A():
        dut._log.info("pulsing reset_ptr_A")
        dut.reset_ptr_A.value = 1
        await tick()
        dut.reset_ptr_A.value = 0
        await tick()

    async def assert_reset_ptr_B():
        dut._log.info("pulsing reset_ptr_B")
        dut.reset_ptr_B.value = 1
        await tick()
        dut.reset_ptr_B.value = 0
        await tick()

    async def assert_reset_ptr_weight():
        dut._log.info("pulsing reset_ptr_weight")
        dut.reset_ptr_weight.value = 1
        await tick()
        dut.reset_ptr_weight.value = 0
        await tick()

    async def assert_reset_ptr_bias():
        dut._log.info("pulsing reset_ptr_bias")
        dut.reset_ptr_bias.value = 1
        await tick()
        dut.reset_ptr_bias.value = 0
        await tick()
        
    
    async def counter():
            dut.counter.value
            await tick()
    
    async def run_test():
        """Run the test"""
        # Reset the DUT
        await reset_dut()
        # layer 1 !!!!
        dut._log.info("layer 1 computing now!!!")
        for i in range(8):
            for i in range(64):
                # Test with read_weights, read_inputs, write_outputs all set to 1
                await apply_inputs(0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0)
                dut._log.info("loading values from memory")
                await tick()

                await assert_bias()

                await assert_start()
                dut._log.info("asserting start")
                while dut.sta_done_computing.value == 0:
                    await tick()
                await apply_inputs(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)
                dut._log.info("asserting done")
                while dut.idle.value == 0:
                    await tick()
                dut._log.info("idle is high")  
                await deassert_done()
                await assert_reset_sta()
                
                # await assert_incr_bias_ptr()
                await assert_incr_input_ptr()
            
            await assert_incr_bias_ptr()
            await assert_incr_weight_ptr()
            await assert_reset_ptr_A()
        # output channel s
        for i in range(16):
            # input channel 1
            for i in range(16):
                # tiling
                # Test with read_weights, read_inputs, write_outputs all set to 1
                await apply_inputs(0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0)
                dut._log.info("loading values from memory")
                await tick()

                await assert_bias()

                await assert_start()
                dut._log.info("asserting start")
                while dut.sta_done_computing.value == 0:
                    await tick()
                await apply_inputs(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)
                dut._log.info("asserting done")
                while dut.idle.value == 0:
                    await tick()
                dut._log.info("idle is high")  
                await deassert_done()
                await assert_reset_sta()
                
                # await assert_incr_bias_ptr()
                await assert_incr_input_ptr()
            # load next weights
            await assert_incr_weight_ptr()
            for i in range(16):
                # tiling
                # Test with read_weights, read_inputs, write_outputs all set to 1
                await apply_inputs(0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0)
                dut._log.info("loading values from memory")
                await tick()

                await assert_bias()

                await assert_start()
                dut._log.info("asserting start")
                while dut.sta_done_computing.value == 0:
                    await tick()
                await apply_inputs(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)
                dut._log.info("asserting done")
                while dut.idle.value == 0:
                    await tick()
                dut._log.info("idle is high")  
                await deassert_done()
                await assert_reset_sta()
                
                # await assert_incr_bias_ptr()
                await assert_incr_input_ptr()
            # input channel 1
            await assert_incr_weight_ptr()
            for i in range(16):
                # tiling
                # Test with read_weights, read_inputs, write_outputs all set to 1
                await apply_inputs(0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0)
                dut._log.info("loading values from memory")
                await tick()

                await assert_bias()

                await assert_start()
                dut._log.info("asserting start")
                while dut.sta_done_computing.value == 0:
                    await tick()
                await apply_inputs(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)
                dut._log.info("asserting done")
                while dut.idle.value == 0:
                    await tick()
                dut._log.info("idle is high")  
                await deassert_done()
                await assert_reset_sta()
                
                # await assert_incr_bias_ptr()
                await assert_incr_input_ptr()
            # load next weights
            await assert_incr_weight_ptr()
            for i in range(16):
                # tiling
                # Test with read_weights, read_inputs, write_outputs all set to 1
                await apply_inputs(0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0)
                dut._log.info("loading values from memory")
                await tick()

                await assert_bias()

                await assert_start()
                dut._log.info("asserting start")
                while dut.sta_done_computing.value == 0:
                    await tick()
                await apply_inputs(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)
                dut._log.info("asserting done")
                while dut.idle.value == 0:
                    await tick()
                dut._log.info("idle is high")  
                await deassert_done()
                await assert_reset_sta()
                
                # await assert_incr_bias_ptr()
                await assert_incr_input_ptr()
            # input channel 1
            await assert_incr_weight_ptr()
            for i in range(16):
                # tiling
                # Test with read_weights, read_inputs, write_outputs all set to 1
                await apply_inputs(0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0)
                dut._log.info("loading values from memory")
                await tick()

                await assert_bias()

                await assert_start()
                dut._log.info("asserting start")
                while dut.sta_done_computing.value == 0:
                    await tick()
                await apply_inputs(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)
                dut._log.info("asserting done")
                while dut.idle.value == 0:
                    await tick()
                dut._log.info("idle is high")  
                await deassert_done()
                await assert_reset_sta()
                
                # await assert_incr_bias_ptr()
                await assert_incr_input_ptr()
            # load next weights
            await assert_incr_weight_ptr()
            for i in range(16):
                # tiling
                # Test with read_weights, read_inputs, write_outputs all set to 1
                await apply_inputs(0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0)
                dut._log.info("loading values from memory")
                await tick()

                await assert_bias()

                await assert_start()
                dut._log.info("asserting start")
                while dut.sta_done_computing.value == 0:
                    await tick()
                await apply_inputs(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)
                dut._log.info("asserting done")
                while dut.idle.value == 0:
                    await tick()
                dut._log.info("idle is high")  
                await deassert_done()
                await assert_reset_sta()
                
                # await assert_incr_bias_ptr()
                await assert_incr_input_ptr()
            # input channel 1
            await assert_incr_weight_ptr()
            for i in range(16):
                # tiling
                # Test with read_weights, read_inputs, write_outputs all set to 1
                await apply_inputs(0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0)
                dut._log.info("loading values from memory")
                await tick()

                await assert_bias()

                await assert_start()
                dut._log.info("asserting start")
                while dut.sta_done_computing.value == 0:
                    await tick()
                await apply_inputs(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)
                dut._log.info("asserting done")
                while dut.idle.value == 0:
                    await tick()
                dut._log.info("idle is high")  
                await deassert_done()
                await assert_reset_sta()
                
                # await assert_incr_bias_ptr()
                await assert_incr_input_ptr()
            # load next weights
            await assert_incr_weight_ptr()
            for i in range(16):
                # tiling
                # Test with read_weights, read_inputs, write_outputs all set to 1
                await apply_inputs(0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0)
                dut._log.info("loading values from memory")
                await tick()

                await assert_bias()

                await assert_start()
                dut._log.info("asserting start")
                while dut.sta_done_computing.value == 0:
                    await tick()
                await apply_inputs(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1)
                dut._log.info("asserting done")
                while dut.idle.value == 0:
                    await tick()
                dut._log.info("idle is high")  
                await deassert_done()
                await assert_reset_sta()
                
                # await assert_incr_bias_ptr()
                await assert_incr_input_ptr()
            # load new bias
            #load new weights
            # reset back to first input channel
        
        # layer 2 !!!!
        
        # Reset the DUT again
        await reset_dut()
    await run_test()
        