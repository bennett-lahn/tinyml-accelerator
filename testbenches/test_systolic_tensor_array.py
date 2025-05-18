import cocotb
import logging
from cocotb.clock import Clock
from cocotb.triggers import RisingEdge
import random

N = 4
VECTOR_WIDTH = 4

@cocotb.test()
async def test_systolic_tensor_array(dut):
    """Test the fixed-size 4×4 STA using 1-D unpacked ports. Does not account for edge pipeline delays created by sliding window."""
    dut._log.setLevel(logging.INFO)
    cocotb.start_soon(Clock(dut.clk, 2, units="ns").start())

    async def tick():
        await RisingEdge(dut.clk)

    async def monitor_inputs_and_outputs(dut):
        cycle = 0
        while True: # Runs indefinitely, or can be modified
            await tick()
            log_lines = []
            for i in range(N):
                a_port = getattr(dut, f"A{i}")
                b_port = getattr(dut, f"B{i}")
                c_port = getattr(dut, f"C{i}")

                a_row = [int(a_port[k].value.signed_integer) for k in range(VECTOR_WIDTH)]
                b_row = [int(b_port[k].value.signed_integer) for k in range(VECTOR_WIDTH)]
                c_row = [int(c_port[j].value.signed_integer) for j in range(N)]
                log_lines.append(f"MON clk={cycle:2d}: A{i}={a_row}  B{i}={b_row}  C{i}={c_row}")
            for line in log_lines:
                dut._log.info(line)
            cycle += 1
    # monitor_task = cocotb.start_soon(monitor_inputs_and_outputs(dut))

    # a and b are lists of inputs for the entire systolic array
    def expected_dot(a, b):
        return sum(int(a[k]) * int(b[k]) for k in range(VECTOR_WIDTH))

    def check_output(label, expected_C_matrix):
        for i in range(N):
            c_port = getattr(dut, f"C{i}")
            for j in range(N):
                out_val = c_port[j].value.signed_integer
                assert out_val == expected_C_matrix[i][j], \
                    f"❌ {label} C[{i}][{j}] = {out_val}, expected {expected_C_matrix[i][j]}"

    async def apply_inputs_drive_only(A, B, load_sum_mat, load_bias_mat, bias_mat, do_reset, stall=False):
        """Drives inputs for the current cycle without advancing the clock."""
        dut.reset.value = 1 if do_reset else 0
        dut.stall.value = 1 if stall and not do_reset else 0

        for i in range(N):
            a_input_val = A[i] if not do_reset else [0]*VECTOR_WIDTH
            b_input_val = B[i] if not do_reset else [0]*VECTOR_WIDTH
            
            getattr(dut, f"A{i}").value = a_input_val
            getattr(dut, f"B{i}").value = b_input_val
            getattr(dut, f"load_sum{i}").value = load_sum_mat[i]
            getattr(dut, f"load_bias{i}").value= load_bias_mat[i]
            getattr(dut, f"bias{i}").value = bias_mat[i]

    async def apply_inputs_and_tick(A, B, load_sum_mat, load_bias_mat, bias_mat, do_reset, stall=False, cycles=1):
        """Apply inputs and advance the clock by inputted number of cycles."""
        await apply_inputs_drive_only(A, B, load_sum_mat, load_bias_mat, bias_mat, do_reset, stall)
        for _ in range(cycles):
            await tick()
        # If reset was asserted, de-assert it after the first tick
        if do_reset:
            dut.reset.value = 0

    # --- Helper Data Structures ---
    zeros_A_B_row   = [0]*VECTOR_WIDTH
    zeros_A_B_mat   = [list(zeros_A_B_row) for _ in range(N)]
    zeros_ctrl_row  = [0]*N
    zeros_ctrl_mat  = [list(zeros_ctrl_row) for _ in range(N)]
    zeros_C_mat     = [[0]*N for _ in range(N)]

    async def run_test_case(A_input_mat, B_input_mat, label=""):
        # 1) Reset
        await apply_inputs_drive_only(zeros_A_B_mat, zeros_A_B_mat, zeros_ctrl_mat, zeros_ctrl_mat, zeros_ctrl_mat, do_reset=True)
        await tick() # Assert reset for one cycle
        await apply_inputs_drive_only(zeros_A_B_mat, zeros_A_B_mat, zeros_ctrl_mat, zeros_ctrl_mat, zeros_ctrl_mat, do_reset=False) # De-assert reset
        await tick() # DUT operates for one cycle with reset low
        check_output(f"{label} Reset State", zeros_C_mat)

        # 2) MAC Operation
        await apply_inputs_drive_only(A_input_mat, B_input_mat, zeros_ctrl_mat, zeros_ctrl_mat, zeros_ctrl_mat, do_reset=False)
        await tick()

        # Matrix stores the expected accumulator values for each PE
        expected_C_state = [[0 for _ in range(N)] for _ in range(N)]

        # Max latency for any PE[i][j] to produce its first result (dot(A[i],B[j])) is max(i,j) + 1 cycles
        # Need to simulate for (N-1) + (N-1) + 1 = 2*N - 1 cycles to see all results.
        max_compute_latency = (N - 1) + (N - 1) + 1 
        for k_cycle in range(max_compute_latency):
            await tick()
            # Update expected_C_state based on which PEs complete their MAC for A_input_mat, B_input_mat
            for r_pe in range(N):
                for c_pe in range(N):
                    # PE[r_pe][c_pe] gets dot(A_input_mat[r_pe], B_input_mat[c_pe]) into its accumulator
                    # at cycle k_cycle = max(r_pe, c_pe).
                    if k_cycle >= max(r_pe, c_pe):
                        expected_C_state[r_pe][c_pe] += expected_dot(A_input_mat[r_pe], B_input_mat[c_pe])
            check_output(f"{label} MAC Cycle k={k_cycle}", expected_C_state)
            
        await apply_inputs_drive_only(zeros_A_B_mat, zeros_A_B_mat, zeros_ctrl_mat, zeros_ctrl_mat, zeros_ctrl_mat, do_reset=False)

        # 3) Final Reset
        await apply_inputs_drive_only(zeros_A_B_mat, zeros_A_B_mat, zeros_ctrl_mat, zeros_ctrl_mat, zeros_ctrl_mat, do_reset=True)
        await tick()
        await apply_inputs_drive_only(zeros_A_B_mat, zeros_A_B_mat, zeros_ctrl_mat, zeros_ctrl_mat, zeros_ctrl_mat, do_reset=False)
        await tick()
        check_output(f"{label} Second Reset State", zeros_C_mat)

    # --- Edge Cases ---
    edge_cases = [
        ([[0]*VECTOR_WIDTH for _ in range(N)], [[0]*VECTOR_WIDTH for _ in range(N)], "Zeros"),
        ([[127]*VECTOR_WIDTH for _ in range(N)], [[1]*VECTOR_WIDTH for _ in range(N)], "Max A"),
        ([[1]*VECTOR_WIDTH for _ in range(N)], [[127]*VECTOR_WIDTH for _ in range(N)], "Max B"),
        ([[-128]*VECTOR_WIDTH for _ in range(N)], [[1]*VECTOR_WIDTH for _ in range(N)], "Min A"),
        ([[1]*VECTOR_WIDTH for _ in range(N)], [[-128]*VECTOR_WIDTH for _ in range(N)], "Min B"),
        ([[127,-128,127,-128] for _ in range(N)], [[-128,127,-128,127] for _ in range(N)], "Mixed extremes"),
    ]
    dut._log.info("🚀 Running STA edge-case tests...")
    for A, B, name in edge_cases:
        await run_test_case(A, B, f"[Edge: {name}]")
    dut._log.info("✅ Edge-case tests passed!")

    # --- Random Tests ---
    dut._log.info("🚀 Running STA random tests...")
    for idx in range(100): # Number of random tests
        A_rand = [[random.randint(-128,127) for _ in range(VECTOR_WIDTH)] for _ in range(N)]
        B_rand = [[random.randint(-128,127) for _ in range(VECTOR_WIDTH)] for _ in range(N)]
        await run_test_case(A_rand, B_rand, f"[Random {idx+1}]")
    dut._log.info("✅ Randomized tests passed!")

    # --- BIAS-LOADING TESTS ---
    dut._log.info("🚀 Running STA bias-loading tests...")
    ones_ctrl_row  = [1]*N
    ones_ctrl_mat  = [list(ones_ctrl_row) for _ in range(N)]
    bias_val_mat = [[random.randint(-100,100) for _ in range(N)] for _ in range(N)]

    # Reset
    await apply_inputs_and_tick(zeros_A_B_mat, zeros_A_B_mat, zeros_ctrl_mat, zeros_ctrl_mat, bias_val_mat, do_reset=True, cycles=2)
    check_output("Bias-Only Reset Check", zeros_C_mat)

    # Load bias into every PE
    await apply_inputs_and_tick(zeros_A_B_mat, zeros_A_B_mat, zeros_ctrl_mat, ones_ctrl_mat, bias_val_mat, do_reset=False, cycles=2)
    # After 1st tick: load_bias is active, bias_val is at PE.sum_in. PE.acc loads it.
    # After 2nd tick: PE.acc holds bias_val. C output reflects it.
    check_output("Bias-Only Load", bias_val_mat)

    # Verify no change with zero-MAC
    dut._log.info("Bias-Only: Verifying no change with zero-MAC...")
    await apply_inputs_and_tick(zeros_A_B_mat, zeros_A_B_mat, zeros_ctrl_mat, zeros_ctrl_mat, bias_val_mat, do_reset=False, cycles=1)
    await tick()
    check_output("Bias + Zero-MAC", bias_val_mat)
    dut._log.info("✅ Bias-Only tests passed!")

    # Load bias + one MAC with non-zero data
    dut._log.info("🚀 Running STA bias + MAC tests...")
    A_mac = [[random.randint(-10,10) for _ in range(VECTOR_WIDTH)] for _ in range(N)]
    B_mac = [[random.randint(-10,10) for _ in range(VECTOR_WIDTH)] for _ in range(N)]
    bias_mac = [[random.randint(-50,50) for _ in range(N)] for _ in range(N)]
    
    expected_after_bias_load = [row[:] for row in bias_mac] # Deep copy
    expected_after_full_mac_wave = [[0]*N for _ in range(N)]
    for r in range(N):
        for c in range(N):
            expected_after_full_mac_wave[r][c] = bias_mac[r][c] + expected_dot(A_mac[r], B_mac[c])

    # Reset
    await apply_inputs_and_tick(zeros_A_B_mat, zeros_A_B_mat, zeros_ctrl_mat, zeros_ctrl_mat, bias_mac, do_reset=True, cycles=2)
    check_output("Bias+MAC Reset Check", zeros_C_mat)

    # Load new bias
    await apply_inputs_and_tick(zeros_A_B_mat, zeros_A_B_mat, zeros_ctrl_mat, ones_ctrl_mat, bias_mac, do_reset=False, cycles=2)
    check_output("Bias+MAC Load", expected_after_bias_load)

    # One MAC step - apply A_mac, B_mac inputs. load_bias should be zero.
    await apply_inputs_drive_only(A_mac, B_mac, zeros_ctrl_mat, zeros_ctrl_mat, bias_mac, do_reset=False)
    await tick()
    
    current_C_state_bias_mac = [row[:] for row in expected_after_bias_load] # Start with loaded biases
    max_compute_latency = (N - 1) + (N - 1) + 1
    for k_cycle in range(max_compute_latency):
        await tick()
        for r_pe in range(N):
            for c_pe in range(N):
                if k_cycle >= max(r_pe, c_pe): # MAC result for this PE is ready
                    # The dot product is added to the bias that was already in the accumulator
                    current_C_state_bias_mac[r_pe][c_pe] += expected_dot(A_mac[r_pe], B_mac[c_pe])
        check_output(f"Bias+MAC Cycle k={k_cycle}", current_C_state_bias_mac)
    dut._log.info("✅ Bias + MAC test passed!")

    # Partial-sum feature currently not tested.

    dut._log.info("✅ All STA tests passed")
