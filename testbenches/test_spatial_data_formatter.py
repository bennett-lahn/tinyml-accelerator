import cocotb, logging
cocotb.log.setLevel(logging.DEBUG)
from cocotb.triggers import RisingEdge
from cocotb.clock import Clock
import random


@cocotb.test()
async def test_spatial_data_formatter(dut):
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
    
    async def assign_input_values():
        dut.patch_pe00_in.value = 0
        dut.patch_pe01_in.value = 1
        dut.patch_pe02_in.value = 2
        dut.patch_pe03_in.value = 3
        dut.patch_pe04_in.value = 4
        dut.patch_pe05_in.value = 5
        dut.patch_pe06_in.value = 6
        dut.patch_pe10_in.value = 10
        dut.patch_pe11_in.value = 11
        dut.patch_pe12_in.value = 12
        dut.patch_pe13_in.value = 13
        dut.patch_pe14_in.value = 14
        dut.patch_pe15_in.value = 15
        dut.patch_pe16_in.value = 16
        dut.patch_pe20_in.value = 20
        dut.patch_pe21_in.value = 21
        dut.patch_pe22_in.value = 22
        dut.patch_pe23_in.value = 23
        dut.patch_pe24_in.value = 24
        dut.patch_pe25_in.value = 25
        dut.patch_pe26_in.value = 26
        dut.patch_pe30_in.value = 30
        dut.patch_pe31_in.value = 31
        dut.patch_pe32_in.value = 32
        dut.patch_pe33_in.value = 33
        dut.patch_pe34_in.value = 34
        dut.patch_pe35_in.value = 35
        dut.patch_pe36_in.value = 36
        dut.patch_pe40_in.value = 40
        dut.patch_pe41_in.value = 41
        dut.patch_pe42_in.value = 42
        dut.patch_pe43_in.value = 43
        dut.patch_pe44_in.value = 44
        dut.patch_pe45_in.value = 45
        dut.patch_pe46_in.value = 46
        dut.patch_pe50_in.value = 50
        dut.patch_pe51_in.value = 51
        dut.patch_pe52_in.value = 52
        dut.patch_pe53_in.value = 53
        dut.patch_pe54_in.value = 54
        dut.patch_pe55_in.value = 55
        dut.patch_pe56_in.value = 56
        dut.patch_pe60_in.value = 60
        dut.patch_pe61_in.value = 61
        dut.patch_pe62_in.value = 62
        dut.patch_pe63_in.value = 63
        dut.patch_pe64_in.value = 64
        dut.patch_pe65_in.value = 65
        dut.patch_pe66_in.value = 66
    
    async def assert_start_formatting():
        """Assert start_formatting signal"""
        dut.start_formatting.value = 1
        await tick()
        dut.start_formatting.value = 0
        await tick()

    async def trace_output():
        """Trace the output"""
        dut._log.info(f"formatted_A0[0]: {dut.formatted_A0[0].value}")
        dut._log.info(f"formatted_A0[1]: {dut.formatted_A0[1].value}")
        dut._log.info(f"formatted_A0[2]: {dut.formatted_A0[2].value}")
        dut._log.info(f"formatted_A0[3]: {dut.formatted_A0[3].value}")
        dut._log.info(f"formatted_A1[0]: {dut.formatted_A1[0].value}")
        dut._log.info(f"formatted_A1[1]: {dut.formatted_A1[1].value}")
        dut._log.info(f"formatted_A1[2]: {dut.formatted_A1[2].value}")
        dut._log.info(f"formatted_A1[3]: {dut.formatted_A1[3].value}")
        dut._log.info(f"formatted_A2[0]: {dut.formatted_A2[0].value}")
        dut._log.info(f"formatted_A2[1]: {dut.formatted_A2[1].value}")
        dut._log.info(f"formatted_A2[2]: {dut.formatted_A2[2].value}")
        dut._log.info(f"formatted_A2[3]: {dut.formatted_A2[3].value}")
        dut._log.info(f"formatted_A3[0]: {dut.formatted_A3[0].value}")
        dut._log.info(f"formatted_A3[1]: {dut.formatted_A3[1].value}")
        dut._log.info(f"formatted_A3[2]: {dut.formatted_A3[2].value}")
        dut._log.info(f"formatted_A3[3]: {dut.formatted_A3[3].value}")
        dut._log.info(f"formatted_data_valid: {dut.formatted_data_valid.value}")
    
    async def run_test():
        """Run the test"""
        await tick()
        await reset_dut()
        await tick()
        await assign_input_values()
        await tick()
        dut.patches_valid.value = 1
        await tick()
        await assert_start_formatting()
        await tick()
        while dut.all_cols_sent.value == 0:
            await trace_output()
            await tick()
        await tick()
        await tick()
        await assert_start_formatting()
        while dut.all_cols_sent.value == 0:
            await tick()
        
        
    await run_test()