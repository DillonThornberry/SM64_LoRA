import json
import re

'''
Step 3: Preprocess data by removing newline chars

* Data used from wiki_text.json (generated in getwikitext.py)
* Preprocessed and stored in wiki_clean.json

'''

# Function to clean newline characters from string values
def clean_newlines(value):
    if isinstance(value, str):
        # Replace multiple newlines with a single newline
        return re.sub(r'\n+', '\n', value)
    return value

jsonFile = open('wiki_text.json', 'r', encoding='utf-8')
data = json.load(jsonFile)


# Clean newlines in each value of the dictionary
cleaned_data = {key: clean_newlines(value) for key, value in data.items()}

# Write the cleaned data back to JSON
with open('wiki_clean.json', 'w', encoding='utf-8') as json_file:
    json.dump(cleaned_data, json_file, indent=4)