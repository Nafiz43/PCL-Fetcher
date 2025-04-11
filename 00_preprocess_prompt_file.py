# Read the input file
with open("data/PCL_Questions_V5_CoT.txt", "r", encoding="utf-8") as file:
    lines = file.readlines()

# Enclose each line in double quotes
modified_lines = [f'"{line.strip()}"\n' for line in lines]

# Write the modified lines to a new file
with open("data/PCL_Questions_V5_CoT.csv", "w", encoding="utf-8") as file:
    file.writelines(modified_lines)

print("Processing complete! Check data dir")
