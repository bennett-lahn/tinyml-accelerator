# Python script to generate a .hex file for a 16x16x8 image tensor
# Each line contains exactly 128 bits (16 bytes) of data
# Data layout: row-major channel-last order [row, col, channel]

IMG_W = 16
IMG_H = 16 
NUM_CHANNELS = 8
FILENAME = "image_data.hex"
BYTES_PER_LINE = 16  # 128 bits = 16 bytes

def generate_hex_file(width, height, channels, filename):
    """
    Generates a .hex file with a test pattern for a 16x16x8 tensor.
    Each line contains exactly 128 bits (16 bytes) of data.
    Uses row-major channel-last layout: data[row][col][channel]
    Each pixel value increments sequentially from 0-255, then wraps.
    """
    total_pixels = width * height * channels
    
    print(f"Generating {width}x{height}x{channels} tensor...")
    print(f"Total pixels: {total_pixels}")
    print(f"Bytes per pixel: 1 (8-bit values)")
    print(f"Bytes per line: {BYTES_PER_LINE} (128 bits)")
    print(f"Expected lines: {(total_pixels + BYTES_PER_LINE - 1) // BYTES_PER_LINE}")
    
    # Generate pixel data in row-major channel-last order
    pixel_data = []
    pixel_value = 0
    
    for row in range(height):
        for col in range(width):
            for ch in range(channels):
                pixel_data.append(pixel_value & 0xFF)  # Keep values 0-255
                pixel_value += 1
    
    # Group pixels into 128-bit lines (16 bytes per line)
    lines = []
    for i in range(0, len(pixel_data), BYTES_PER_LINE):
        hex_line = ""
        for j in range(BYTES_PER_LINE):
            if i + j < len(pixel_data):
                byte_val = pixel_data[i + j]
                hex_line += f"{byte_val:02x}"
            else:
                # Pad incomplete lines with zeros
                hex_line += "00"
        lines.append(hex_line)
    
    # Write to file
    try:
        with open(filename, 'w') as f:
            for line in lines:
                f.write(line + "\n")
        
        print(f"Successfully generated '{filename}' with {len(lines)} lines.")
        
        # Verification: show data layout
        print("\nData layout verification (row-major channel-last):")
        print("First few pixels (row, col, channel) -> value:")
        for row in range(min(2, height)):
            for col in range(min(2, width)):
                for ch in range(channels):
                    pixel_idx = (row * width + col) * channels + ch
                    if pixel_idx < len(pixel_data):
                        print(f"  ({row:2d},{col:2d},{ch}) -> 0x{pixel_data[pixel_idx]:02x} ({pixel_data[pixel_idx]:3d})")
        
        print(f"\nFirst line (bytes 0-15): {lines[0]}")
        print(f"Second line (bytes 16-31): {lines[1]}")
        
        # Show how to interpret a 32-bit word from the data
        print(f"\nExample 32-bit words from first line:")
        first_line = lines[0]
        for i in range(0, min(16, len(first_line)), 8):  # 4 bytes = 8 hex chars
            word = first_line[i:i+8]
            if len(word) == 8:
                # Convert to little-endian interpretation
                bytes_le = [word[j:j+2] for j in range(6, -1, -2)]
                print(f"  Word {i//8}: 0x{word} -> LE: 0x{''.join(bytes_le)}")
                
    except IOError as e:
        print(f"Error writing to file '{filename}': {e}")

if __name__ == "__main__":
    generate_hex_file(IMG_W, IMG_H, NUM_CHANNELS, FILENAME)
