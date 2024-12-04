import requests
from bs4 import BeautifulSoup
import time
import json

'''
Step 2 in data collection process: Getting text from links

* Wiki links are found in links.txt
* Links were found by getlinks.py
* Raw text stored in wiki_text.json

'''

BASE_URL = "https://ukikipedia.net"

# Function to fetch and scrape text from a single wiki page
def scrape_wiki_page(url):
    print(f"Scraping: {url}")
    response = requests.get(BASE_URL + url)
    
    # Ensure a successful response
    if response.status_code != 200:
        print(f"Failed to retrieve {url}")
        return ""
    
    # Parse the page with BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract the content of the page
    content_div = soup.find('div', class_='mw-parser-output')
    if content_div:
        # Extract and return all text from the content div
        return content_div.get_text(separator="\n").strip()
    else:
        print("no content div")
    return ""



txt = open("links.txt", 'r', encoding='utf-8')
lines = txt.read().splitlines()

output = {}

for line in lines:
    text = scrape_wiki_page(line)
    output[line] = text

with open('wiki_text.json', 'w', encoding='utf-8') as json_file:
    # Write the dictionary to the file in JSON format
    json.dump(output, json_file, indent=4)




