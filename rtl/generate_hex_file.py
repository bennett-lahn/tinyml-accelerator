# Python script to generate a .hex file for an image
# with 8-bit words and dimensions IMG_W x IMG_H.

IMG_W = 96
IMG_H = 96
FILENAME = "image_data.hex"

def generate_hex_file(width, height, filename):
    """
    Generates a .hex file with a test pattern.
    Each line contains one 32-bit value that represents 4 8-bit pixels.
    The pattern is: [pixel3, pixel2, pixel1, pixel0] where each pixel is different
    """
    total_pixels = width * height
    lines = []
    current_value = 0

    blocks_x = width // 4
    blocks_y = height // 4
    for by in range(blocks_y):
        for bx in range(blocks_x):
            hex_value = ""
            # Process one 4x4 block (16 pixels => 128 bits)
            # Each row in the block: pixel0 | pixel1 | pixel2 | pixel3
            for row in range(4):
                for col in range(4):
                    pixel = current_value & 0xFF
                    hex_value += format(pixel, '02X')
                    current_value = (current_value + 1) % 256
            lines.append(hex_value)

    try:
        with open(filename, 'w') as f:
            for line in lines:
                f.write(line + "\n")
        print(f"Successfully generated '{filename}' with {total_pixels} lines.")
    except IOError as e:
        print(f"Error writing to file '{filename}': {e}")

if __name__ == "__main__":
    generate_hex_file(IMG_W, IMG_H, FILENAME)
