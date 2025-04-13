"""
Developed at DECAL Lab in CS Department @ UC Davis by Nafiz Imtiaz Khan (nikhan@ucdavis.edu)
Copyright © 2025 The Regents of the University of California, Davis campus. All Rights Reserved. Used with permission.
"""
# Read the input file
with open("data/PCL_Questions_V5_CoT.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()

# Enclose each line in double quotes
modified_lines = [f'"{line.strip()}"\n' for line in lines]

# Write the modified lines to a new file
with open("data/PCL_Questions_V5_CoT.csv", "w", encoding="utf-8") as file:
    file.writelines(modified_lines)

print("Processing complete! Check data dir")
