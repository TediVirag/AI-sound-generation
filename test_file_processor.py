import os
import shutil
import random
import string
from pathlib import Path
from datetime import datetime

def generate_unique_id(length=8, existing_ids=None):
    """Generate a unique alphanumeric ID."""
    if existing_ids is None:
        existing_ids = set()
    
    while True:
        # Generate random ID with letters and numbers
        id_chars = string.ascii_letters + string.digits
        unique_id = ''.join(random.choice(id_chars) for _ in range(length))
        
        if unique_id not in existing_ids:
            existing_ids.add(unique_id)
            return unique_id

def process_files_recursively(input_folder, output_path, id_length=8):
    """
    Process files recursively:
    1. Add unique ID to end of original filename
    2. Create copy in output path with only the ID as filename
    3. Generate pairing document
    """
    
    # Convert to Path objects for easier handling
    input_path = Path(input_folder)
    output_path = Path(output_path)
    
    # Validate input folder exists
    if not input_path.exists():
        print(f"Error: Input folder '{input_folder}' does not exist.")
        return
    
    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Track used IDs to ensure uniqueness
    used_ids = set()
    
    # List to store pairing information
    pairings = []
    
    # Counter for processed files
    processed_count = 0
    
    print(f"Processing files from: {input_path.absolute()}")
    print(f"Output path: {output_path.absolute()}")
    print("-" * 60)
    
    # Walk through all files recursively
    for root, dirs, files in os.walk(input_path):
        for filename in files:
            original_file_path = Path(root) / filename
            
            # Skip if it's a directory or hidden file
            if original_file_path.is_dir() or filename.startswith('.'):
                continue
            
            try:
                # Generate unique ID
                unique_id = generate_unique_id(id_length, used_ids)
                
                # Get file extension
                file_extension = original_file_path.suffix
                
                # Create new filename with ID at the end
                name_without_ext = original_file_path.stem
                new_filename_with_id = f"{name_without_ext}_{unique_id}{file_extension}"
                
                # Create new file path in the same directory
                new_file_path = original_file_path.parent / new_filename_with_id
                
                # Rename original file
                original_file_path.rename(new_file_path)
                
                # Create copy in output folder with only ID as name
                id_only_filename = f"{unique_id}{file_extension}"
                output_file_path = output_path / id_only_filename
                
                # Copy file to output folder
                shutil.copy2(new_file_path, output_file_path)
                
                # Store pairing information
                pairings.append((new_filename_with_id, id_only_filename))
                
                processed_count += 1
                
                print(f"✓ Processed: {filename}")
                print(f"  → Renamed to: {new_filename_with_id}")
                print(f"  → Copy created: {id_only_filename}")
                print(f"  → ID: {unique_id}")
                print()
                
            except Exception as e:
                print(f"✗ Error processing {filename}: {str(e)}")
                print()
    
    # Generate pairing document
    if pairings:
        pairing_file_path = output_path / "_file_pairings.txt"
        try:
            with open(pairing_file_path, 'w', encoding='utf-8') as f:
                # Write header
                f.write("FILE PAIRING DOCUMENT\n")
                f.write("=" * 50 + "\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}\n")
                f.write(f"Input folder: {input_path.absolute()}\n")
                f.write(f"Output folder: {output_path.absolute()}\n")
                f.write(f"Total files processed: {processed_count}\n")
                f.write("=" * 50 + "\n\n")
                
                # Write column headers
                f.write("ORIGINAL FILENAME (with ID)".ljust(40) + " → " + "ID-ONLY FILENAME\n")
                f.write("-" * 80 + "\n")
                
                # Write pairings
                for original_with_id, id_only in sorted(pairings):
                    f.write(f"{original_with_id.ljust(40)} → {id_only}\n")
                
                # Write footer
                f.write("\n" + "=" * 50 + "\n")
                f.write("Legend:\n")
                f.write("• Left column: Original filename with unique ID appended\n")
                f.write("• Right column: Copy filename (ID only) in output folder\n")
            
            print(f"📄 Pairing document created: {pairing_file_path}")
            
        except Exception as e:
            print(f"✗ Error creating pairing document: {str(e)}")
    
    print("-" * 60)
    print(f"Processing complete! {processed_count} files processed.")
    print(f"Original files renamed with unique IDs in their original locations.")
    print(f"Copies with ID-only names created in: {output_path.absolute()}")

def validate_path(path_str, path_type="path"):
    """Validate and expand path string."""
    if not path_str.strip():
        return None
    
    # Expand user home directory (~) and environment variables
    expanded_path = os.path.expanduser(os.path.expandvars(path_str.strip()))
    return Path(expanded_path)

def main():
    """Main function with user input."""
    print("=== Enhanced File Processor with Unique IDs ===\n")
    
    # Get input folder from user
    while True:
        input_folder_str = input("Enter the folder path to process: ").strip()
        input_path = validate_path(input_folder_str)
        
        if input_path and input_path.exists():
            break
        else:
            print(f"Error: Path '{input_folder_str}' does not exist or is invalid. Please try again.\n")
    
    # Get output path (full path, not just folder name)
    while True:
        output_path_str = input("Enter the full output path (e.g., /path/to/output or C:\\output): ").strip()
        if not output_path_str:
            output_path_str = input("Output path cannot be empty. Please enter a valid path: ").strip()
        
        output_path = validate_path(output_path_str)
        if output_path:
            # Ask for confirmation if path exists and is not empty
            if output_path.exists() and any(output_path.iterdir()):
                confirm = input(f"Output path '{output_path}' exists and is not empty. Continue? (y/N): ").strip().lower()
                if confirm not in ['y', 'yes']:
                    continue
            break
        else:
            print(f"Error: Invalid path '{output_path_str}'. Please try again.\n")
    
    # Get ID length (optional)
    while True:
        id_length_input = input("Enter ID length (default: 8, min: 4, max: 16): ").strip()
        try:
            if not id_length_input:
                id_length = 8
                break
            
            id_length = int(id_length_input)
            if 4 <= id_length <= 16:
                break
            else:
                print("ID length must be between 4 and 16 characters.")
        except ValueError:
            print("Please enter a valid number.")
    
    print(f"\n" + "=" * 60)
    print(f"CONFIGURATION SUMMARY")
    print(f"=" * 60)
    print(f"Input folder:  {input_path.absolute()}")
    print(f"Output path:   {output_path.absolute()}")
    print(f"ID length:     {id_length} characters")
    print(f"Pairing file:  {output_path.absolute() / 'file_pairings.txt'}")
    print(f"=" * 60)
    
    # Confirm before processing
    confirm = input("\nProceed with processing? (y/N): ").strip().lower()
    if confirm in ['y', 'yes']:
        print(f"\nStarting processing...\n")
        process_files_recursively(str(input_path), str(output_path), id_length)
    else:
        print("Processing cancelled.")

def process_files_with_params(input_folder, output_path, id_length=8):
    """
    Direct function call for programmatic use.
    
    Args:
        input_folder (str): Path to folder to process
        output_path (str): Full path to output directory
        id_length (int): Length of generated IDs (default: 8)
    """
    return process_files_recursively(input_folder, output_path, id_length)

if __name__ == "__main__":
    main()