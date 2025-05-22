import cocotb
import logging
from cocotb.clock      import Clock
from cocotb.triggers   import RisingEdge
import random

@cocotb.test()
async def test_maxpool_unit_streaming(dut):
    """Drive rows and sample outputs each cycle to catch mid-input pooled values."""
    SA_N     = len(dut.in_valid)
    FILTER_H = int(dut.FILTER_H.value)
    FILTER_W = int(dut.FILTER_W.value)
    # number of pooled blocks = (SA_N/FILTER_H)*(SA_N/FILTER_W)
    num_blocks = (SA_N // FILTER_H) * (SA_N // FILTER_W)

    # start clock
    cocotb.start_soon(Clock(dut.clk, 10, units="ns").start())
    dut._log.setLevel(logging.INFO)

    async def tick():
        """Advance one clock."""
        await RisingEdge(dut.clk)

    async def reset_dut():
        dut.reset.value = 1
        await tick()
        dut.reset.value = 0
        await tick()

    def compute_expected(mat, base_r, base_c):
        out = []
        nb_r = SA_N // FILTER_H
        nb_c = SA_N // FILTER_W
        for br in range(nb_r):
            for bc in range(nb_c):
                maxv = None
                for pr in range(FILTER_H):
                    for pc in range(FILTER_W):
                        r = br*FILTER_H + pr
                        c = bc*FILTER_W + pc
                        v = mat[r][c]
                        if maxv is None or v > maxv:
                            maxv = v
                out_r = base_r + br*FILTER_H
                out_c = base_c + bc*FILTER_W
                out.append((out_r, out_c, maxv))
        return out

    async def drive_and_capture(mat, base_r, base_c):
        seen = []
        # drive each row, sample outputs immediately
        # deassert valids, then start
        dut.pos_row.value = base_r
        dut.pos_col.value = base_c
        for ch in range(SA_N):
            dut.in_valid[ch].value = 0
        await tick()
        for row in range(SA_N):
            # assert valids with this row
            for ch in range(SA_N):
                dut.in_valid[ch].value = 1
                dut.in_row[ch].value   = base_r + row
                dut.in_col[ch].value   = base_c + ch
                dut.in_data[ch].value  = mat[row][ch]
            await tick()

            # sample possible output
            if dut.out_valid.value.integer:
                seen.append((
                    dut.out_row .value.integer,
                    dut.out_col .value.integer,
                    dut.out_data.value.signed_integer
                ))
        for ch in range(SA_N):
             dut.in_valid[ch].value = 0
        # flush remaining outputs
        for _ in range(num_blocks):
            await tick()
            if dut.out_valid.value.integer:
                seen.append((
                    dut.out_row .value.integer,
                    dut.out_col .value.integer,
                    dut.out_data.value.signed_integer
                ))
        return seen

    # 1) Basic fixed‐matrix test
    dut._log.info("🚀 Basic fixed‐matrix streaming test")
    mat0 = [
        [ 1,  5,  3,  2],
        [ 4,  8,  6,  7],
        [-1,  0, -2, -3],
        [10, 12,  9, 11]
    ]
    await reset_dut()
    exp0 = compute_expected(mat0, 0, 0)
    seen0 = await drive_and_capture(mat0, 0, 0)
    assert seen0 == exp0, f"Mismatch: saw {seen0}, expected {exp0}"
    dut._log.info("✅ Basic test passed")

    # 2) Randomized tests
    dut._log.info("🚀 Randomized streaming tests")
    for idx in range(10):
        mat = [[random.randint(-128,127) for _ in range(SA_N)]
               for _ in range(SA_N)]
        await reset_dut()
        rand_row = random.randint(0,511)
        rand_col = random.randint(0,511)
        exp = compute_expected(mat, rand_row, rand_col)
        seen = await drive_and_capture(mat, rand_row, rand_col)
        assert seen == exp, f"[rnd {idx}] saw {seen}, exp {exp}"
        dut._log.info(f"✅ Random test {idx} passed")

    # 3) Offset base_row/base_col
    dut._log.info("🚀 Offset base_row/base_col streaming test")
    base_r, base_c = 2, 1
    size = SA_N + max(base_r, base_c)
    matB = [[random.randint(-50,50) for _ in range(size)]
            for _ in range(size)]
    await reset_dut()
    expB = compute_expected(matB, base_r, base_c)
    seenB = await drive_and_capture(matB, base_r, base_c)
    assert seenB == expB, f"Offset saw {seenB}, exp {expB}"
    dut._log.info("✅ Offset test passed")

    dut._log.info("✅ All streaming maxpool tests passed")
