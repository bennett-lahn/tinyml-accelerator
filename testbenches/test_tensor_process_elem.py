import cocotb, logging
cocotb.log.setLevel(logging.DEBUG)
from cocotb.triggers import RisingEdge
from cocotb.clock import Clock
import random

@cocotb.test()
async def test_tensor_process_elem(dut):
    """Tensor processing element test with random and edge cases"""
    CLOCK_PERIOD = 2
    dut._log.setLevel(logging.DEBUG)
    # Start clock
    cocotb.start_soon(Clock(dut.clk, CLOCK_PERIOD, units="ns").start())

    # Advance one clock cycle
    async def tick():
        await RisingEdge(dut.clk)

    # Update inputs to dut
    async def apply_inputs(a, b, sum_in, load_sum, reset):
        dut.reset.value = reset
        dut.load_sum.value = load_sum
        dut.sum_in.value = sum_in
        for i in range(4):
            dut.left_in[i].value = a[i]
            dut.top_in[i].value = b[i]
        await tick()
        await tick()

    # Calculate the expected dot product for 4 element vector
    def expected_dot(a, b):
        return sum(int(a[i]) * int(b[i]) for i in range(4))

    # Check if value from dut matches expected value, print appropriate message
    def check_equal(name, actual, expected):
            assert actual == expected, f"❌ {name} FAILED: Got {actual}, Expected {expected}"
            dut._log.info(f"✅ {name} PASSED")

    # Runs test case using provided dut input values
    async def run_test_case(a, b, load_val, label=""):
        # Reset
        await apply_inputs(a, b, 0, False, True)
        check_equal(f"{label} Reset sum_out", dut.sum_out.value.signed_integer, 0)

        # Load sum
        await apply_inputs(a, b, load_val, True, False)
        check_equal(f"{label} Load sum_out", dut.sum_out.value.signed_integer, load_val)

        # MAC
        await apply_inputs(a, b, 0, False, False)
        expected = load_val + expected_dot(a, b)
        check_equal(f"{label} MAC accumulation", dut.sum_out.value.signed_integer, expected)

        # Second MAC
        await tick()
        expected += expected_dot(a, b)
        check_equal(f"{label} Second MAC accumulation", dut.sum_out.value.signed_integer, expected)

        # Final reset
        await apply_inputs(a, b, 0, False, True)
        check_equal(f"{label} Second reset", dut.sum_out.value.signed_integer, 0)

    dut._log.info("🚀 Starting tensor_process_elem test with edge cases and random values...")

    edge_cases = [
        ([0, 0, 0, 0], [0, 0, 0, 0], 0, "Zero inputs"),
        ([127, 127, 127, 127], [1, 1, 1, 1], 0, "Max positive A"),
        ([1, 1, 1, 1], [127, 127, 127, 127], 0, "Max positive B"),
        ([-128, -128, -128, -128], [1, 1, 1, 1], 0, "Max negative A"),
        ([1, 1, 1, 1], [-128, -128, -128, -128], 0, "Max negative B"),
        ([127, -128, 127, -128], [-128, 127, -128, 127], 1000, "Mixed extremes"),
    ]

    for a, b, sum_in, label in edge_cases:
        await run_test_case(a, b, sum_in, f"[Edge: {label}]")

    for i in range(5):
        a = [random.randint(-128, 127) for _ in range(4)]
        b = [random.randint(-128, 127) for _ in range(4)]
        load_val = random.randint(-1000, 1000)
        await run_test_case(a, b, load_val, f"[Random {i + 1}]")

    dut._log.info("✅ All edge and randomized tests completed successfully!")
