filepath = r'd:\IIT\Final Year Project\FishSense\dashboard\app.py'

with open(filepath, 'r', encoding='utf-8') as f:
    lines = f.readlines()

expander_idx = None
for i, line in enumerate(lines):
    if 'Choose Your Fishing Location' in line and 'st.expander' in line:
        expander_idx = i
        break

if expander_idx is None:
    print("ERROR: Could not find expander line")
    exit(1)

print(f"Found expander at line {expander_idx + 1}")

end_idx = None
for i in range(expander_idx + 1, len(lines)):
    if '# Initialize session state for fetched data' in lines[i]:
        end_idx = i
        break

if end_idx is None:
    print("ERROR: Could not find end of expander block")
    exit(1)

print(f"Expander content: lines {expander_idx + 2} to {end_idx}")

for i in range(expander_idx + 1, end_idx):
    line = lines[i]
    if line.strip() == '':
        continue
    lines[i] = '    ' + line

with open(filepath, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("SUCCESS")
