import json

# Step 1: Load JSON from a file
with open('wiki_clean.json', 'r') as file:
    data = json.load(file)

output = []

for key in data:
    prompt = key.split('/')[-1]
    prompt = " ".join(prompt.split('_'))
    prompt = " - ".join(prompt.split('#'))
    prompt = "'".join(prompt.split('%27'))
    prompt = "Explain " + prompt

    answer = data[key]
    
    answer = " ".join(answer.split('\n'))

    if answer.startswith("This article is a  stub"):
        answer = answer[98:]

    line = {"prompt": prompt, "completion": answer}
    output.append(line)

    #print(answer, '\n\n')



# Step 3: Save the updated JSON to a file
# Path to the output .jsonl file
output_file = "sm64.jsonl"

# Writing the list to a JSONL file
with open(output_file, 'w') as f:
    for item in output:
        json.dump(item, f)
        f.write("\n")  # Ensure each dictionary is written on a new line

print(f"Data has been written to {output_file}")