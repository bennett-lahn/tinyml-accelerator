import cocotb
import logging
import random
from cocotb.triggers import Timer

def to_signed(value, bits):
    """Convert Python int to a signed integer of 'bits' width (2's complement)."""
    mask = (1 << bits) - 1
    val = value & mask
    if (val >> (bits - 1)) & 1: # Check sign bit
        return val - (1 << bits)
    return val

def py_multiply_by_quantized_multiplier(val_in, multiplier_in, shift_in):
    """
    Python model of the MultiplyByQuantizedMultiplier function from SystemVerilog.
    Assumes SystemVerilog '>>>' on a signed type is an arithmetic shift as per IEEE 1800-2017.
    """
    # val_in, multiplier_in are signed 32-bit. shift_in is signed 6-bit.

    prod = val_in * multiplier_in # Python handles large integers
    rounded_64 = prod + (1 << 30)
    shifted_val_intermediate = rounded_64 >> 31
    tmp = to_signed(shifted_val_intermediate, 32)
    if shift_in > 0: # Arithmetic Right shift
        result = tmp >> shift_in
    elif shift_in < 0: # Arithmetic Left shift (same as logical left shift)
        result = tmp << -shift_in
        # Result is int32_t, so it's truncated/wrapped to 32 bits.
        result = to_signed(result, 32)
    else: # shift_in == 0
        result = tmp
    
    return to_signed(result, 32) # Ensure final result is proper 32-bit signed

def py_requantize_activate(dut, acc, quant_mult, shift, choose_zero_point, qmin, qmax):
    """Python model of the entire requantize_activate_unit logic."""
    
    norm_zero_point = -128
    special_zero_point = -1

    scaled = py_multiply_by_quantized_multiplier(acc, quant_mult, shift)
    zero_point_to_add = special_zero_point if choose_zero_point else norm_zero_point
    with_zp = to_signed(scaled + zero_point_to_add, 32)
    
    # Clamp to [QMIN, QMAX] (implements quantized ReLU6)
    # and handle final conversion to int8_t behavior
    if with_zp < qmin:
        clamped_val = qmin
    elif with_zp > qmax:
        clamped_val = qmax
    else:
        clamped_val = to_signed(with_zp, 8)
    return clamped_val

@cocotb.test()
async def test_requantize_activate_unit(dut):
    """Test for the requantize_activate_unit module."""
    dut._log.setLevel(logging.INFO)

    # Get QMIN and QMAX from DUT parameters
    try:
        QMIN = int(dut.QMIN.value)
        QMAX = int(dut.QMAX.value)
    except AttributeError:
        dut._log.error("FATAL: Could not read QMIN/QMAX parameters from DUT. Ensure they are accessible.")
        QMIN = -128 # Fallback, but test might not be accurate
        QMAX = 127  # Fallback
        # return # Or raise an error to stop the test
    except ValueError:
        dut._log.error("FATAL: QMIN/QMAX from DUT are not valid integers.")
        QMIN = -128 
        QMAX = 127
        # return

    dut._log.info(f"DUT Parameters: QMIN={QMIN}, QMAX={QMAX}")
    
    async def apply_and_check(test_name, acc_val, qm_val, sh_val, czp_val):
        """Helper coroutine to set inputs, get expected value, and check DUT output."""
        dut.acc.value = acc_val
        dut.quant_mult.value = qm_val
        dut.shift.value = sh_val
        dut.choose_zero_point.value = czp_val

        expected_out = py_requantize_activate(dut, acc_val, qm_val, sh_val, czp_val, QMIN, QMAX)

        await Timer(1,units='ps')

        dut_out_val = dut.out.value.signed_integer

        assert dut_out_val == expected_out, \
            (f"❌ {test_name} FAILED: acc={acc_val}, qm={qm_val}, sh={sh_val}, czp={czp_val}. "
             f"DUT_out={dut_out_val}, Expected={expected_out}")
        dut._log.debug(f"✅ {test_name} PASSED: acc={acc_val}, qm={qm_val}, sh={sh_val}, czp={czp_val} -> out={dut_out_val}")

    # --- Directed Test Cases ---
    dut._log.info("🚀 Starting Directed Test Cases...")

    dut.acc.value = 0
    dut.quant_mult.value = 0
    dut.shift.value = 0
    dut.choose_zero_point.value = 0
    await Timer(1,units='ps')

    # Test Case 1: Zero accumulator
    await apply_and_check("Zero Acc", acc_val=0, qm_val=1073741824, sh_val=0, czp_val=0) # QM = 2^30

    # Test Case 2: Simple positive scaling, no shift, norm_zp
    # acc=100, QM=2^30 (scale=0.5 if total right shift is 31), shift=0. Expected scaled = 50.
    # with_zp = 50 + (-128) = -78
    await apply_and_check("Positive Scale, Norm ZP", acc_val=100, qm_val=(1<<30), sh_val=0, czp_val=0)

    # Test Case 3: Positive scaling, right shift, norm_zp
    # acc=200, QM=2^30, shift=1 (effective total right shift 31+1=32). Expected scaled = 200 * 2^30 / 2^32 = 50.
    # with_zp = 50 + (-128) = -78
    await apply_and_check("Positive Scale, Right Shift, Norm ZP", acc_val=200, qm_val=(1<<30), sh_val=1, czp_val=0)

    # Test Case 4: Positive scaling, left shift, norm_zp
    # acc=25, QM=2^30, shift=-1 (effective total right shift 31-1=30). Expected scaled = 25 * 2^30 / 2^30 = 25.
    # with_zp = 25 + (-128) = -103
    await apply_and_check("Positive Scale, Left Shift, Norm ZP", acc_val=25, qm_val=(1<<30), sh_val=-1, czp_val=0)

    # Test Case 5: Negative accumulator
    # acc=-100, QM=2^30, shift=0. Expected scaled = -50.
    # with_zp = -50 + (-128) = -178. Clamped to QMIN (-128).
    await apply_and_check("Negative Acc, Norm ZP, Clamp Low", acc_val=-100, qm_val=(1<<30), sh_val=0, czp_val=0)

    # Test Case 6: Special zero point
    # acc=260, QM=2^30, shift=0. Expected scaled = 130.
    # with_zp = 130 + (-1) = 129. Clamped to QMAX (127).
    await apply_and_check("Positive Scale, Special ZP, Clamp High", acc_val=260, qm_val=(1<<30), sh_val=0, czp_val=1)
    
    # Test Case 7: Value that results in QMIN
    # acc=-300, QM=2^30, shift=0. Scaled = -150. with_zp = -150 + (-128) = -278. Clamped = -128
    await apply_and_check("Clamp to QMIN", acc_val=-300, qm_val=(1<<30), sh_val=0, czp_val=0)

    # Test Case 8: Value that results in QMAX
    # acc=510, QM=2^30, shift=0. Scaled = 255. with_zp = 255 + (-128) = 127. Clamped = 127
    await apply_and_check("Clamp to QMAX", acc_val=510, qm_val=(1<<30), sh_val=0, czp_val=0)

    # Test Case 9: Value that stays in range (close to zero after ZP)
    # acc=256, QM=2^30, shift=0. Scaled = 128. with_zp = 128 + (-128) = 0.
    await apply_and_check("In Range (output 0)", acc_val=256, qm_val=(1<<30), sh_val=0, czp_val=0)

    # Test Case 10: Max shift right
    # acc=1000, QM=2^30, shift=31. Scaled = 1000 * 2^30 / 2^(31+31) -> very small, likely 0.
    # with_zp = 0 + (-128) = -128.
    await apply_and_check("Max Right Shift", acc_val=1000, qm_val=(1<<30), sh_val=31, czp_val=0)

    # Test Case 11: Max shift left (careful with overflow in intermediate `tmp << -shift`)
    # acc=1, QM=1, shift=-31. tmp after initial scaling (near 0). 0 << 31 = 0.
    # with_zp = 0 + (-128) = -128.
    await apply_and_check("Max Left Shift", acc_val=1, qm_val=1, sh_val=-31, czp_val=0)
    
    # Test Case 12: Large positive quant_mult
    await apply_and_check("Large Pos QM", acc_val=10, qm_val=(1<<31)-1, sh_val=5, czp_val=0)
    
    # Test Case 13: Large negative quant_mult
    await apply_and_check("Large Neg QM", acc_val=10, qm_val=-(1<<31), sh_val=5, czp_val=0)

    dut._log.info("✅ Directed Test Cases PASSED")

    # --- Randomized Test Case ---
    dut._log.info("🚀 Starting Randomized Test Case...")
    num_random_tests = 1000
    # dut._log.setLevel(logging.DEBUG) # Uncomment for very verbose random logs

    # Define reasonable ranges for random values, can be tuned
    # Full int32 range for acc and quant_mult can lead to very large intermediate products.
    acc_min, acc_max = -(1<<31), (1<<31)-1 # Full range for acc
    qm_min, qm_max = -(1<<31), (1<<31)-1    # Full range for quant_mult
    # qm_min, qm_max = (1<<20), (1<<31)-1 # Example: positive large quant_mult
    
    shift_min, shift_max = -32, 31 # 6-bit signed range

    for i in range(num_random_tests):
        rand_acc = random.randint(acc_min, acc_max)
        rand_qm = random.randint(qm_min, qm_max)
        rand_sh = random.randint(shift_min, shift_max)
        rand_czp = random.randint(0, 1)
        
        # dut._log.debug(f"Random Test {i+1}/{num_random_tests}: acc={rand_acc}, qm={rand_qm}, sh={rand_sh}, czp={rand_czp}")
        await apply_and_check(f"Random Test {i+1}", rand_acc, rand_qm, rand_sh, rand_czp)
        if i % (num_random_tests // 10) == 0 and i > 0:
             dut._log.info(f"  Random test progress: {i}/{num_random_tests}")


    dut._log.info("✅ Randomized Test Case PASSED")
    # dut._log.setLevel(logging.INFO) # Reset log level if changed
    dut._log.info("✅ All test_requantize_activate_unit tests completed.")

