import cocotb
from cocotb.clock    import Clock
from cocotb.triggers import RisingEdge, ReadOnly
import random

N            = 4
VECTOR_WIDTH = 4

@cocotb.test()
async def test_systolic_tensor_array_fixed(dut):
    """Test the fixed-size 4×4 STA using 1-D unpacked ports."""
    # Start the clock
    cocotb.start_soon(Clock(dut.clk, 2, units="ns").start())

    async def tick():
        await RisingEdge(dut.clk)

    async def monitor_inputs_and_outputs(dut, num_cycles=None):
        """
        Monitor A0..A3, B0..B3, and C0..C3 on each rising edge.
        If num_cycles is None, runs forever; else stops after num_cycles.
        """
        cycle = 0

        while num_cycles is None or cycle < num_cycles:
            # wait a clock and let all signals settle
            await RisingEdge(dut.clk)
            await ReadOnly()

            for i in range(N):
                # grab the 1-D ports by name
                a_port = getattr(dut, f"A{i}")
                b_port = getattr(dut, f"B{i}")
                c_port = getattr(dut, f"C{i}")

                # build Python lists of lane values
                a_row = [int(a_port[k].value.signed_integer)
                         for k in range(VECTOR_WIDTH)]
                b_row = [int(b_port[k].value.signed_integer)
                         for k in range(VECTOR_WIDTH)]
                c_row = [int(c_port[j].value.signed_integer)
                         for j in range(N)]

                dut._log.info(f"MON clk={cycle:2d}: "
                              f"A{i}={a_row}  "
                              f"B{i}={b_row}  "
                              f"C{i}={c_row}")

            cycle += 1

        dut._log.info(f"MONITOR: stopped after {cycle} cycles")

    cocotb.start_soon(monitor_inputs_and_outputs(dut))

    def expected_dot(a, b):
        return sum(int(a[k]) * int(b[k]) for k in range(VECTOR_WIDTH))

    def check_output(label, expected):
        for i in range(N):
            c_port = getattr(dut, f"C{i}")
            for j in range(N):
                out = c_port[j].value.signed_integer
                assert out == expected[i][j], \
                    f"❌ {label}[{i}][{j}] = {out}, expected {expected[i][j]}"
        dut._log.info(f"✅ {label} PASSED")

    async def apply_inputs(A, B,
                           load_sum_mat, load_bias_mat, bias_mat,
                           do_reset, cycles=1):
        # 1) Drive reset if requested
        if (do_reset):
            dut.reset.value = int(do_reset)
            for _ in range(1):
                await tick()
            dut.reset.value = 0

        # 2) Drive all ports
        for i in range(N):
            getattr(dut, f"A{i}").value       = A[i]
            getattr(dut, f"B{i}").value       = B[i]
            getattr(dut, f"load_sum{i}").value = load_sum_mat[i]
            getattr(dut, f"load_bias{i}").value= load_bias_mat[i]
            getattr(dut, f"bias{i}").value     = bias_mat[i]

        # 3) Let pipeline flops latch inputs
        for _ in range(cycles):
            await tick()

    async def run_test_case(A, B, label=""):
        # compute golden NxN matrix multiply of 4-wide dot products
        expected = [[expected_dot(A[i], B[j]) for j in range(N)]
                    for i in range(N)]

        zeros_row = [0]*VECTOR_WIDTH
        zeros_sum = [[0]*N for _ in range(N)]
        zeros_bias= [[0]*N for _ in range(N)]

        # 1) Reset
        await apply_inputs(A, B, zeros_sum, zeros_sum, zeros_bias,
                           do_reset=True,  cycles=1)
        check_output(f"{label} reset", [[0]*N for _ in range(N)])

        # 2) MAC
        await apply_inputs(A, B, zeros_sum, zeros_sum, zeros_bias,
                           do_reset=False, cycles=2)
        check_output(f"{label} First cycle", expected)

        # 3) Reset
        await apply_inputs(A, B, zeros_sum, zeros_sum, zeros_bias,
                           do_reset=True, cycles=1)
        check_output(f"{label} Second reset", [[0]*N for _ in range(N)])


    # edge cases
    edge_cases = [
        ([[  0]*VECTOR_WIDTH for _ in range(N)],
         [[  0]*VECTOR_WIDTH for _ in range(N)], "Zeros"),

        ([[127]*VECTOR_WIDTH for _ in range(N)],
         [[  1]*VECTOR_WIDTH for _ in range(N)], "Max A"),

        ([[  1]*VECTOR_WIDTH for _ in range(N)],
         [[127]*VECTOR_WIDTH for _ in range(N)], "Max B"),

        ([[-128]*VECTOR_WIDTH for _ in range(N)],
         [[  1]*VECTOR_WIDTH for _ in range(N)], "Min A"),

        ([[  1]*VECTOR_WIDTH for _ in range(N)],
         [[-128]*VECTOR_WIDTH for _ in range(N)], "Min B"),

        ([[127,-128,127,-128] for _ in range(N)],
         [[-128,127,-128,127] for _ in range(N)], "Mixed extremes"),
    ]

    dut._log.info("🚀 STA edge-case tests")
    for A, B, name in edge_cases:
        await run_test_case(A, B, f"[Edge: {name}]")

    dut._log.info("🚀 STA random tests")
    for idx in range(5):
        A = [[random.randint(-128,127) for _ in range(VECTOR_WIDTH)]
             for _ in range(N)]
        B = [[random.randint(-128,127) for _ in range(VECTOR_WIDTH)]
             for _ in range(N)]
        await run_test_case(A, B, f"[Random {idx+1}]")

    # BIAS‐LOADING TESTS

    dut._log.info("🚀 STA bias‐loading tests")

    # Helper mats
    zeros_A       = [[0]*VECTOR_WIDTH for _ in range(N)]
    zeros_B       = [[0]*VECTOR_WIDTH for _ in range(N)]
    zeros_sum     = [[0]*N for _ in range(N)]
    zeros_bias_ls = [[0]*N for _ in range(N)]
    ones_bias_ls  = [[1]*N for _ in range(N)]

    # Load‐only (no MAC)
    bias_only = [[random.randint(-100,100) for _ in range(N)] for _ in range(N)]

    # Reset
    await apply_inputs(zeros_A, zeros_B,
                       zeros_sum, zeros_bias_ls, bias_only,
                       do_reset=True,  cycles=1)
    check_output("Bias-Only reset", [[0]*N for _ in range(N)])

    # Load bias into every PE
    await apply_inputs(zeros_A, zeros_B,
                       zeros_sum, ones_bias_ls, bias_only,
                       do_reset=False, cycles=2)
    check_output("Bias‐Only load", bias_only)

    # Verify no change with zero‐MAC
    await apply_inputs(zeros_A, zeros_B,
                       zeros_sum, zeros_bias_ls, bias_only,
                       do_reset=False, cycles=2)
    check_output("Bias + zero‐MAC", bias_only)

    # Load bias + one MAC with non‐zero data
    # Create random A,B
    A_rand = [[random.randint(-10,10) for _ in range(VECTOR_WIDTH)] for _ in range(N)]
    B_rand = [[random.randint(-10,10) for _ in range(VECTOR_WIDTH)] for _ in range(N)]
    bias_rand = [[random.randint(-50,50) for _ in range(N)] for _ in range(N)]
    # Golden dot product
    golden_dot = lambda a,b: sum(int(a[k])*int(b[k]) for k in range(VECTOR_WIDTH))
    expected_bias      = bias_rand
    expected_after_mac = [
      [ bias_rand[i][j] + golden_dot(A_rand[i], B_rand[j])
        for j in range(N) ]
      for i in range(N)
    ]

    # Reset
    await apply_inputs(A_rand, B_rand,
                       zeros_sum, zeros_bias_ls, bias_rand,
                       do_reset=True,  cycles=1)
    check_output("Bias-Only reset", [[0]*N for _ in range(N)])
    # Load new bias
    await apply_inputs(A_rand, B_rand,
                       zeros_sum, ones_bias_ls, bias_rand,
                       do_reset=False, cycles=2)
    check_output("Bias‐Rand load", expected_bias)

    # One MAC step
    await apply_inputs(A_rand, B_rand,
                       zeros_sum, zeros_bias_ls, bias_rand,
                       do_reset=False, cycles=2)
    check_output("Bias + MAC", expected_after_mac)

    dut._log.info("✅ All bias‐loading tests passed!")

    # PARTIAL‐SUM LOADING TESTS

    dut._log.info("🚀 STA partial‐sum loading tests")

    # Helper zero mats
    zeros_A    = [[0]*VECTOR_WIDTH for _ in range(N)]
    zeros_B    = [[0]*VECTOR_WIDTH for _ in range(N)]
    zeros_sum  = [[0]*N            for _ in range(N)]
    zeros_bias = [[0]*N            for _ in range(N)]

    # Do a first MAC on a fresh random A/B to get C0
    A_ps = [[random.randint(-10,10) for _ in range(VECTOR_WIDTH)] for _ in range(N)]
    B_ps = [[random.randint(-10,10) for _ in range(VECTOR_WIDTH)] for _ in range(N)]
    # Golden row-0
    golden0 = [ expected_dot(A_ps[0], B_ps[j]) for j in range(N) ]
    # Build the full expected matrix after first MAC:
    # Only row0 is filled; rows 1–3 remain zero
    expected_first = [ golden0 ] + [[0]*N]*3

    # Reset + first MAC
    await apply_inputs(A_ps, B_ps, zeros_sum, zeros_bias, zeros_bias,
                       do_reset=True,  cycles=1)
    check_output("PS first reset", [[0]*N for _ in range(N)])
    await apply_inputs(A_ps, B_ps, zeros_sum, zeros_bias, zeros_bias,
                       do_reset=False, cycles=2)
    check_output("PS first MAC", expected_first)

    # Load_sum into row1: no new MAC since A/B=0; row1 should pick up row0’s results
    load_sum_ps = [ row.copy() for row in zeros_sum ]
    load_sum_ps[1] = [1]*N    # only row 1 asserts load_sum

    await apply_inputs(zeros_A, zeros_B, load_sum_ps, zeros_bias, zeros_bias,
                       do_reset=False, cycles=2)
    expected_row1 = [ golden0, golden0, [0]*N, [0]*N ]
    check_output("PS inject row1", expected_row1)

    # Chain into row2
    load_sum_ps = [ row.copy() for row in zeros_sum ]
    load_sum_ps[2] = [1]*N    # only row 2 asserts load_sum

    await apply_inputs(zeros_A, zeros_B, load_sum_ps, zeros_bias, zeros_bias,
                       do_reset=False, cycles=2)
    expected_row2 = [ golden0, golden0, golden0, [0]*N ]
    check_output("PS inject row2", expected_row2)

    dut._log.info("✅ All partial-sum loading tests passed!")

    dut._log.info("✅ All STA tests passed")
