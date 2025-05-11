import requests
from bs4 import BeautifulSoup
from pathlib import Path
import re
import time

# let's create a set of locals referring to our directory and working directory 
home = Path.home()
work_dir = (home / 'housing_project')
data = (work_dir / 'data')
minutes = (data / 'meeting_minutes')
minutes_raw = (minutes / 'pdfs')
minutes_clean = (minutes / 'processed')
code = Path.cwd() 

# Define the URL of the archive page
ARCHIVE_URL = "https://sfplanning.org/cpc-hearing-archives"

# Define the local directory to save PDFs
OUTPUT_DIR = minutes_raw
OUTPUT_DIR.mkdir(exist_ok=True)

# Set up headers to mimic a browser visit
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

def fetch_archive_page(url):
    """Fetches the content of the archive page."""
    response = requests.get(url, headers=HEADERS)
    response.raise_for_status()
    return response.text

def parse_minutes_links(html_content):
    """Parses the HTML content and extracts all 'Minutes' PDF links."""
    soup = BeautifulSoup(html_content, "html.parser")
    minutes_links = []

    # Find all anchor tags that contain the text 'Minutes'
    for a_tag in soup.find_all("a", string=re.compile(r"Minutes", re.IGNORECASE)):
        href = a_tag.get("href")
        if href and href.lower().endswith(".pdf"):
            # Construct full URL if the href is relative
            if not href.startswith("http"):
                href = f"https://sfplanning.org{href}"
            minutes_links.append(href)

    return minutes_links

def download_pdfs(links, output_dir):
    """Downloads each PDF from the list of links into the specified directory."""
    for idx, link in enumerate(links, start=1):
        try:
            response = requests.get(link, headers=HEADERS)
            response.raise_for_status()

            # Extract the filename from the URL
            filename = link.split("/")[-1]
            file_path = output_dir / filename

            # Save the PDF content to a file
            with open(file_path, "wb") as f:
                f.write(response.content)

            print(f"[{idx}/{len(links)}] Downloaded: {filename}")

            # Be polite and avoid overwhelming the server
            time.sleep(1)

        except requests.exceptions.RequestException as e:
            print(f"Failed to download {link}: {e}")

def main():
    print("Fetching archive page...")
    html_content = fetch_archive_page(ARCHIVE_URL)

    print("Parsing 'Minutes' PDF links...")
    minutes_links = parse_minutes_links(html_content)

    print(f"Found {len(minutes_links)} 'Minutes' PDFs. Starting download...")
    download_pdfs(minutes_links, OUTPUT_DIR)
    print("Download completed.")

if __name__ == "__main__":
    main()
