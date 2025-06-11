import os
import sys

def concatenate_python_files(package_directory, output_filename="concatenated_package_code.txt"):
    """
    Concatenates all Python files within a directory and its subdirectories
    into a single text file with clear separators.

    Args:
        package_directory (str): The path to the root of the Python package.
        output_filename (str): The name of the output text file.
    """
    if not os.path.isdir(package_directory):
        print(f"Error: Directory not found at '{package_directory}'")
        return

    print(f"Processing directory: {package_directory}")
    print(f"Writing to output file: {output_filename}")
    print("-" * 50)

    file_count = 0
    try:
        with open(output_filename, 'w', encoding='utf-8') as outfile:
            for root, _, files in os.walk(package_directory):
                for filename in files:
                    if filename.endswith('.py'):
                        filepath = os.path.join(root, filename)
                        file_count += 1
                        print(f"Including: {filepath}")

                        outfile.write("=" * 50 + "\n")
                        outfile.write(f"{filepath}\n")
                        outfile.write("=" * 50 + "\n")

                        try:
                            with open(filepath, 'r', encoding='utf-8') as infile:
                                content = infile.read()
                                outfile.write(content)
                                # Ensure a newline at the end of the file content if not already present
                                if not content.endswith('\n'):
                                    outfile.write('\n')

                        except UnicodeDecodeError:
                            print(f"Warning: Could not decode file '{filepath}' with utf-8. Skipping content.")
                            outfile.write(f"\n# !! Could not read file content (encoding error) !!\n")
                        except IOError as e:
                            print(f"Warning: Could not read file '{filepath}': {e}")
                            outfile.write(f"\n# !! Could not read file content (IO Error: {e}) !!\n")

                        outfile.write("-" * 50 + "\n")
                        outfile.write("\n") # Add an extra newline for better separation visually

        print("-" * 50)
        print(f"Successfully concatenated {file_count} Python files into '{output_filename}'.")

    except IOError as e:
        print(f"Error: Could not write to output file '{output_filename}': {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python your_script_name.py <path_to_package_directory>")
        sys.exit(1)

    package_path = sys.argv[1]
    concatenate_python_files(package_path)