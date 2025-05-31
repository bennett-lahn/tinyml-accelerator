#!/usr/bin/env python3

import numpy as np

def show_create_test_pattern(width=8, height=8, channels=8):
    """Show the structured test pattern (but this is NOT actually used in the RAM simulation)"""
    print("=== STRUCTURED TEST PATTERN (create_test_pattern) ===")
    print("This pattern is created but NOT actually used in the RAM simulation!")
    print(f"Dimensions: {height}x{width}x{channels//4} (height x width x channel_groups)")
    print("Pattern: val = (row << 4) | (col << 2) | (channel)")
    print()
    
    data = np.zeros((height, width, channels//4), dtype=np.uint32)
    for row in range(height):
        for col in range(width):
            for ch in range(0, channels, 4):  # Process 4 channels at a time
                packed = 0
                for c in range(4):
                    if ch + c < channels:
                        val = (row << 4) | (col << 2) | (ch + c)
                        packed |= (val & 0xFF) << (c * 8)
                data[row][col][ch//4] = packed
    
    # Show first few values
    for row in range(min(4, height)):
        for col in range(min(4, width)):
            for ch_group in range(min(2, channels//4)):
                packed_val = data[row][col][ch_group]
                print(f"data[{row}][{col}][{ch_group}] = 0x{packed_val:08x}")
                
                # Unpack to show individual channel values
                ch0 = (packed_val >>  0) & 0xFF
                ch1 = (packed_val >>  8) & 0xFF  
                ch2 = (packed_val >> 16) & 0xFF
                ch3 = (packed_val >> 24) & 0xFF
                
                base_ch = ch_group * 4
                print(f"  ch{base_ch+0}=0x{ch0:02x} ch{base_ch+1}=0x{ch1:02x} ch{base_ch+2}=0x{ch2:02x} ch{base_ch+3}=0x{ch3:02x}")
                print()

def show_actual_ram_data():
    """Show the actual data pattern used in simulate_ram_reads"""
    print("=== ACTUAL RAM DATA PATTERN (simulate_ram_reads) ===")
    print("This is the data that actually gets fed into the DUT during testing!")
    print("Pattern: base_val = 0x10203040 + (addr % 256)")
    print("  ram_dout0 = base_val")
    print("  ram_dout1 = base_val + 0x01010101") 
    print("  ram_dout2 = base_val + 0x02020202")
    print("  ram_dout3 = base_val + 0x03030303")
    print()
    
    print("Examples for first 16 addresses:")
    for addr in range(16):
        base_val = 0x10203040 + (addr % 256)
        ram_dout0 = base_val
        ram_dout1 = base_val + 0x01010101
        ram_dout2 = base_val + 0x02020202  
        ram_dout3 = base_val + 0x03030303
        
        print(f"addr={addr:2d}: ram_dout0=0x{ram_dout0:08x} ram_dout1=0x{ram_dout1:08x} ram_dout2=0x{ram_dout2:08x} ram_dout3=0x{ram_dout3:08x}")
        
        # Show as individual bytes (channels)
        print(f"        bytes: [0x{(ram_dout0>>0)&0xFF:02x} 0x{(ram_dout0>>8)&0xFF:02x} 0x{(ram_dout0>>16)&0xFF:02x} 0x{(ram_dout0>>24)&0xFF:02x}]" +
              f" [0x{(ram_dout1>>0)&0xFF:02x} 0x{(ram_dout1>>8)&0xFF:02x} 0x{(ram_dout1>>16)&0xFF:02x} 0x{(ram_dout1>>24)&0xFF:02x}]" +
              f" [0x{(ram_dout2>>0)&0xFF:02x} 0x{(ram_dout2>>8)&0xFF:02x} 0x{(ram_dout2>>16)&0xFF:02x} 0x{(ram_dout2>>24)&0xFF:02x}]" +
              f" [0x{(ram_dout3>>0)&0xFF:02x} 0x{(ram_dout3>>8)&0xFF:02x} 0x{(ram_dout3>>16)&0xFF:02x} 0x{(ram_dout3>>24)&0xFF:02x}]")
        print()

def explain_data_flow():
    """Explain how the data flows through the system"""
    print("=== DATA FLOW EXPLANATION ===")
    print("1. The unified_buffer requests data from RAM using ram_addr")
    print("2. The simulate_ram_reads() function responds with 4x 32-bit words")
    print("3. Each 32-bit word contains 4 channels (4 bytes) for the same spatial position")
    print("4. The unified_buffer stores this as buffer_7x7[row][col] = {ram_dout3, ram_dout2, ram_dout1, ram_dout0}")
    print("5. The spatial_data_formatter then extracts specific bytes for each PE:")
    print("   - A0[0] gets the first byte (channel 0)")
    print("   - A0[1] gets the second byte (channel 1)")
    print("   - A0[2] gets the third byte (channel 2)")
    print("   - A0[3] gets the fourth byte (channel 3)")
    print()
    print("So for ram_dout0 = 0x10203040:")
    print("  - A0[0] = 0x40 (bits 7:0)")
    print("  - A0[1] = 0x30 (bits 15:8)")
    print("  - A0[2] = 0x20 (bits 23:16)")
    print("  - A0[3] = 0x10 (bits 31:24)")

if __name__ == "__main__":
    show_create_test_pattern()
    print("\n" + "="*80 + "\n")
    show_actual_ram_data()
    print("\n" + "="*80 + "\n")
    explain_data_flow() 