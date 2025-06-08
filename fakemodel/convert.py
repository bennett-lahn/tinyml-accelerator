# This script requires the Pillow library.
# You can install it using pip:
# pip install Pillow

from PIL import Image
import argparse
import os # Added for path manipulation

def image_to_hex(image_path, output_path, size=(32, 32), pixels_per_line=16):
    """
    Loads an image, converts it to grayscale, resizes it, and saves the
    pixel data to a hex file in row-major order.

    The output format is a single 2D block of hex values, with a specified
    number of pixel values per line. The values represent 8-bit unsigned
    integers (0-255).

    Args:
        image_path (str): The path to the input image file.
        output_path (str): The path where the output hex file will be saved.
        size (tuple): A tuple (width, height) to resize the image to.
        pixels_per_line (int): The number of pixel hex values to write per line.
    """
    try:
        print("--- Script starting ---")
        
        # Open the image using Pillow
        with Image.open(image_path) as img:
            print(f"Successfully opened '{image_path}'")
            
            # --- Image Processing ---
            # 1. Resize the image to the specified dimensions (e.g., 32x32)
            #    We use LANCZOS for a high-quality downscale.
            resized_img = img.resize(size, Image.Resampling.LANCZOS)
            print("Image resized.")

            # 2. Convert the image to grayscale ('L' mode for luminance)
            #    Each pixel will be a single 8-bit value (0-255).
            grayscale_img = resized_img.convert('L')
            print("Image converted to grayscale.")


            # --- Hex Conversion and File Writing ---
            # Get the pixel data from the image
            pixels = grayscale_img.load()
            width, height = grayscale_img.size

            # Get the absolute path for the output file to avoid confusion
            absolute_output_path = os.path.abspath(output_path)
            print(f"Attempting to write output to: {absolute_output_path}")

            # Open the output file in write mode
            with open(output_path, 'w') as f:
                # Write a header commenting the source and dimensions
                f.write(f"# Grayscale hex data for {image_path}\n")
                f.write(f"# Resolution: {width}x{height}\n")
                f.write(f"# Format: Row-major, {pixels_per_line} pixels per line, no spaces.\n")
                f.write("# Each value is an 8-bit integer in hexadecimal.\n\n")

                # --- Write Grayscale Data ---
                f.write(f"# Grayscale Data ({width}x{height})\n")
                
                # Iterate through each row of the image
                for y in range(height):
                    # Get all hex values for the current row as a list of strings
                    full_row_hex = [format(pixels[x, y], '02x') for x in range(width)]
                    
                    # Split the row's hex values into chunks and write each chunk as a new line
                    for i in range(0, width, pixels_per_line):
                        chunk = full_row_hex[i:i + pixels_per_line]
                        # Join without spaces to create a continuous hex string
                        f.write("".join(chunk) + "\n")
                
            print(f"--- Success! ---")
            print(f"Successfully converted '{image_path}' to '{absolute_output_path}'")
            print(f"The output file contains {width * height} hex values with {pixels_per_line} values per line.")

    except FileNotFoundError:
        print(f"--- Error ---")
        print(f"Error: The file '{image_path}' was not found.")
    except Exception as e:
        print(f"--- Error ---")
        print(f"An unexpected error occurred: {e}")

if __name__ == '__main__':
    # --- Command-Line Argument Parsing ---
    # This allows you to run the script from your terminal and pass file paths easily.
    # Example usage:
    # python your_script_name.py my_image.png output.hex
    
    parser = argparse.ArgumentParser(
        description="Convert an image file to a 32x32 grayscale hex file for hardware testing."
    )
    parser.add_argument(
        "input_image",
        type=str,
        help="Path to the input image file (e.g., cat.jpg, icon.png)."
    )
    parser.add_argument(
        "output_file",
        type=str,
        help="Path for the generated hex file (e.g., image_data.hex)."
    )

    args = parser.parse_args()

    # Call the main function with the provided arguments
    image_to_hex(args.input_image, args.output_file)
