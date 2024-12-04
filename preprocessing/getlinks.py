import requests
from bs4 import BeautifulSoup
import time


'''
Step 1 in data collection process - Getting links which data will be scraped from

* Ukikipedia is the main source for Super Mario 64 speedrunning tutorials and info
* This bot crawls through the website and grabs all internal links which could contain useful training info
* Links are written to links.txt to later be scraped by getwikitext.py

'''

BASE_URL = "https://ukikipedia.net"
START_PAGE = "/wiki/Main_Page"
visited_links = set()  # Set to track visited links


# Recursively fetches all relevant links from wiki pages without revisiting links.
def get_all_wiki_links(url, depth=1, max_depth=3):

    # Stop if maximum depth is reached
    if depth > max_depth:
        return
    
    full_url = BASE_URL + url

    # Avoid revisiting the same page
    if url in visited_links:
        return
    visited_links.add(url)
    
    # Fetch the page content
    print(f"Scraping: {full_url} (Depth: {depth})")
    response = requests.get(full_url)
    soup = BeautifulSoup(response.text, "html.parser")
    

    # Find all relevant /wiki/ links on the page
    links = set()
    for link in soup.find_all("a", href=True):
        href = link['href']
        if href.startswith("/wiki/") and "/wiki/edit/" not in href and "/wiki/info/" not in href and "/wiki/history" not in href and ":" not in href[len("/wiki/"):]:
            links.add(href)

    # Recursively visit each new link
    for link in links:
        #time.sleep(.01)  # Throttle in case of overloading the server
        get_all_wiki_links(link, depth + 1, max_depth)


# Start the recursive scraping process
get_all_wiki_links(START_PAGE)
print(len(list(visited_links)))

output = ""
for link in list(visited_links):
    output += link + '\n'

txt = open("links.txt", "w")
print(output, file=txt)
