import os

# Header to insert
header = '''"""
Developed at DECAL Lab in CS Department @ UC Davis by Nafiz Imtiaz Khan (nikhan@ucdavis.edu)
Copyright © 2025 The Regents of the University of California, Davis campus. All Rights Reserved. Used with permission.
"""
'''

def add_header_to_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Skip if header already exists
    if header.strip() in content:
        print(f"✔️ Skipped (already has header): {file_path}")
        return

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(header + "\n" + content)
    print(f"✅ Header added to: {file_path}")

def process_directory(dir_path):
    for root, _, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".py"):
                full_path = os.path.join(root, file)
                add_header_to_file(full_path)

if __name__ == "__main__":
    target_dir = ('/home/nafiz/Documents/PCL').strip()
    if not os.path.isdir(target_dir):
        print("❌ Invalid directory path.")
    else:
        process_directory(target_dir)
