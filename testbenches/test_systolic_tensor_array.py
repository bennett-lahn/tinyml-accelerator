import cocotb
from cocotb.triggers import RisingEdge
from cocotb.clock import Clock
import random


N = 4
VECTOR_WIDTH = 4

@cocotb.test()
async def test_systolic_tensor_array(dut):
    """Test Vsystolic_tensor_array with random and edge case inputs"""

    cocotb.start_soon(Clock(dut.clk, 2, units="ns").start())

    async def tick():
        await RisingEdge(dut.clk)

    def expected_dot(a, b):
        return sum(int(a[i]) * int(b[i]) for i in range(VECTOR_WIDTH))

    def check_output(name, expected):
        for i in range(N):
            for j in range(N):
                actual = int(dut.C_out[i][j].value)
                assert actual == expected[i][j], f"❌ {name}[{i}][{j}] Got {actual}, Expected {expected[i][j]}"
        dut._log.info(f"✅ {name} PASSED")

    async def apply_inputs(A, B, load_sum, reset):
        dut.reset.value = reset
        for i in range(N):
            for v in range(VECTOR_WIDTH):
                dut.A_in[i][v].value = A[i][v]
                dut.B_in[i][v].value = B[i][v]
            for j in range(N):
                dut.load_sum[i][j].value = load_sum[i][j]
        await tick()

    async def run_test_case(A, B, load_sum, label=""):
        expected = [[expected_dot(A[i], B[j]) for j in range(N)] for i in range(N)]

        # Reset
        await apply_inputs(A, B, load_sum, True)
        check_output(f"{label} Reset", [[0] * N for _ in range(N)])

        # Load and compute
        await apply_inputs(A, B, load_sum, False)
        check_output(f"{label} First cycle", expected)

        for _ in range(3):  # Simulate additional cycles
            await tick()
        check_output(f"{label} Final C_out", expected)

        # Second reset
        await apply_inputs(A, B, load_sum, True)
        check_output(f"{label} Second reset", [[0] * N for _ in range(N)])

    dut._log.info("🚀 Starting systolic tensor array test with edge and random cases...")

    edge_cases = [
        ([[0]*VECTOR_WIDTH for _ in range(N)], [[0]*VECTOR_WIDTH for _ in range(N)], "Zeros"),
        ([[127]*VECTOR_WIDTH for _ in range(N)], [[1]*VECTOR_WIDTH for _ in range(N)], "Max A"),
        ([[1]*VECTOR_WIDTH for _ in range(N)], [[127]*VECTOR_WIDTH for _ in range(N)], "Max B"),
        ([[-128]*VECTOR_WIDTH for _ in range(N)], [[1]*VECTOR_WIDTH for _ in range(N)], "Min A"),
        ([[1]*VECTOR_WIDTH for _ in range(N)], [[-128]*VECTOR_WIDTH for _ in range(N)], "Min B"),
        ([[127, -128, 127, -128]]*N, [[-128, 127, -128, 127]]*N, "Mixed extremes")
    ]

    load_sum_diag = [[(i == j) for j in range(N)] for i in range(N)]

    for A, B, label in edge_cases:
        await run_test_case(A, B, load_sum_diag, f"[Edge: {label}]")

    for i in range(5):
        A = [[random.randint(-128, 127) for _ in range(VECTOR_WIDTH)] for _ in range(N)]
        B = [[random.randint(-128, 127) for _ in range(VECTOR_WIDTH)] for _ in range(N)]
        await run_test_case(A, B, load_sum_diag, f"[Random {i+1}]")

    dut._log.info("✅ All systolic array tests completed successfully!")
