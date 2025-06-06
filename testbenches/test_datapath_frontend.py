import cocotb, logging
cocotb.log.setLevel(logging.DEBUG)
from cocotb.triggers import RisingEdge
from cocotb.clock import Clock
import random


@cocotb.test()
async def test_datapath_frontend(dut):
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
    
    async def assert_start_extraction():
        """Assert start_extraction signal"""
        dut.start_extraction.value = 1
        await tick()
        dut.start_extraction.value = 0
        await tick()
        
    async def assert_next_channel_group():
        """Assert next_channel_group signal"""
        dut.next_channel_group.value = 1
        await tick()
        dut.next_channel_group.value = 0
        await tick()
        
    async def assert_next_spatial_block():
        """Assert next_spatial_block signal"""
        dut.next_spatial_block.value = 1
        await tick()
        dut.next_spatial_block.value = 0
        await tick()
    
    async def incr_layer_idx():
        """Increment layer_idx signal"""
        dut.layer_idx.value += 1
        await tick()
    
    async def trace_patch_outputs():
        """Trace all patch PE outputs"""
        dut._log.debug("=== PATCH PE OUTPUTS ===")
        # Row 0
        dut._log.debug(f"patch_pe00_out: 0x{int(dut.patch_pe00_out.value):08x}")
        dut._log.debug(f"patch_pe01_out: 0x{int(dut.patch_pe01_out.value):08x}")
        dut._log.debug(f"patch_pe02_out: 0x{int(dut.patch_pe02_out.value):08x}")
        dut._log.debug(f"patch_pe03_out: 0x{int(dut.patch_pe03_out.value):08x}")
        dut._log.debug(f"patch_pe04_out: 0x{int(dut.patch_pe04_out.value):08x}")
        dut._log.debug(f"patch_pe05_out: 0x{int(dut.patch_pe05_out.value):08x}")
        dut._log.debug(f"patch_pe06_out: 0x{int(dut.patch_pe06_out.value):08x}")
        # Row 1
        dut._log.debug(f"patch_pe10_out: 0x{int(dut.patch_pe10_out.value):08x}")
        dut._log.debug(f"patch_pe11_out: 0x{int(dut.patch_pe11_out.value):08x}")
        dut._log.debug(f"patch_pe12_out: 0x{int(dut.patch_pe12_out.value):08x}")
        dut._log.debug(f"patch_pe13_out: 0x{int(dut.patch_pe13_out.value):08x}")
        dut._log.debug(f"patch_pe14_out: 0x{int(dut.patch_pe14_out.value):08x}")
        dut._log.debug(f"patch_pe15_out: 0x{int(dut.patch_pe15_out.value):08x}")
        dut._log.debug(f"patch_pe16_out: 0x{int(dut.patch_pe16_out.value):08x}")
        # Row 2
        dut._log.debug(f"patch_pe20_out: 0x{int(dut.patch_pe20_out.value):08x}")
        dut._log.debug(f"patch_pe21_out: 0x{int(dut.patch_pe21_out.value):08x}")
        dut._log.debug(f"patch_pe22_out: 0x{int(dut.patch_pe22_out.value):08x}")
        dut._log.debug(f"patch_pe23_out: 0x{int(dut.patch_pe23_out.value):08x}")
        dut._log.debug(f"patch_pe24_out: 0x{int(dut.patch_pe24_out.value):08x}")
        dut._log.debug(f"patch_pe25_out: 0x{int(dut.patch_pe25_out.value):08x}")
        dut._log.debug(f"patch_pe26_out: 0x{int(dut.patch_pe26_out.value):08x}")
        # Row 3
        dut._log.debug(f"patch_pe30_out: 0x{int(dut.patch_pe30_out.value):08x}")
        dut._log.debug(f"patch_pe31_out: 0x{int(dut.patch_pe31_out.value):08x}")
        dut._log.debug(f"patch_pe32_out: 0x{int(dut.patch_pe32_out.value):08x}")
        dut._log.debug(f"patch_pe33_out: 0x{int(dut.patch_pe33_out.value):08x}")
        dut._log.debug(f"patch_pe34_out: 0x{int(dut.patch_pe34_out.value):08x}")
        dut._log.debug(f"patch_pe35_out: 0x{int(dut.patch_pe35_out.value):08x}")
        dut._log.debug(f"patch_pe36_out: 0x{int(dut.patch_pe36_out.value):08x}")
        # Row 4
        dut._log.debug(f"patch_pe40_out: 0x{int(dut.patch_pe40_out.value):08x}")
        dut._log.debug(f"patch_pe41_out: 0x{int(dut.patch_pe41_out.value):08x}")
        dut._log.debug(f"patch_pe42_out: 0x{int(dut.patch_pe42_out.value):08x}")
        dut._log.debug(f"patch_pe43_out: 0x{int(dut.patch_pe43_out.value):08x}")
        dut._log.debug(f"patch_pe44_out: 0x{int(dut.patch_pe44_out.value):08x}")
        dut._log.debug(f"patch_pe45_out: 0x{int(dut.patch_pe45_out.value):08x}")
        dut._log.debug(f"patch_pe46_out: 0x{int(dut.patch_pe46_out.value):08x}")
        # Row 5
        dut._log.debug(f"patch_pe50_out: 0x{int(dut.patch_pe50_out.value):08x}")
        dut._log.debug(f"patch_pe51_out: 0x{int(dut.patch_pe51_out.value):08x}")
        dut._log.debug(f"patch_pe52_out: 0x{int(dut.patch_pe52_out.value):08x}")
        dut._log.debug(f"patch_pe53_out: 0x{int(dut.patch_pe53_out.value):08x}")
        dut._log.debug(f"patch_pe54_out: 0x{int(dut.patch_pe54_out.value):08x}")
        dut._log.debug(f"patch_pe55_out: 0x{int(dut.patch_pe55_out.value):08x}")
        dut._log.debug(f"patch_pe56_out: 0x{int(dut.patch_pe56_out.value):08x}")
        # Row 6
        dut._log.debug(f"patch_pe60_out: 0x{int(dut.patch_pe60_out.value):08x}")
        dut._log.debug(f"patch_pe61_out: 0x{int(dut.patch_pe61_out.value):08x}")
        dut._log.debug(f"patch_pe62_out: 0x{int(dut.patch_pe62_out.value):08x}")
        dut._log.debug(f"patch_pe63_out: 0x{int(dut.patch_pe63_out.value):08x}")
        dut._log.debug(f"patch_pe64_out: 0x{int(dut.patch_pe64_out.value):08x}")
        dut._log.debug(f"patch_pe65_out: 0x{int(dut.patch_pe65_out.value):08x}")
        dut._log.debug(f"patch_pe66_out: 0x{int(dut.patch_pe66_out.value):08x}")
        dut._log.debug(f"patches_valid: {dut.patches_valid.value}")
        dut._log.debug("========================")
    
    async def run_test():
        """Run the test"""
        await tick()
        await reset_dut()
        await tick()
        dut.layer_idx.value = 0
        await assert_start_extraction()
        while dut.all_channels_done.value == 0:
            while dut.patches_valid.value == 0:
                await tick()
            await trace_patch_outputs()
            await tick()
            await assert_next_spatial_block()
            await tick()
        
        # test just the first patch 
    
    await run_test()