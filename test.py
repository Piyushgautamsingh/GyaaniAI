# from langchain_community.document_loaders import SeleniumURLLoader

# loader = SeleniumURLLoader(urls=["https://apisix.apache.org/docs/apisix/getting-started/README"])
# docs = loader.load()

# for doc in docs:
#     print(doc.page_content[:500])  # Truncated for readability
import mechanize
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import os
import re

# Create a folder to save the files
SAVE_DIR = "apisix_docs"
os.makedirs(SAVE_DIR, exist_ok=True)

br = mechanize.Browser()
br.set_handle_robots(False)
br.addheaders = [("User-agent", "Mozilla/5.0 (X11; Linux x86_64)")]

visited = set()

def sanitize_filename(url):
    parsed = urlparse(url)
    path = parsed.path.strip("/")
    path = re.sub(r"[^\w\-_.]", "_", path)
    return path or "index"

def crawl(url, depth=1):
    if depth == 0 or url in visited:
        return

    print(f"Crawling: {url}")
    visited.add(url)

    try:
        response = br.open(url)
        html = response.read()
    except Exception as e:
        print(f"Failed to open {url}: {e}")
        return

    soup = BeautifulSoup(html, "html.parser")
    
    # Extract and save clean text content
    text = soup.get_text(separator="\n", strip=True)
    filename = sanitize_filename(url) + ".txt"
    filepath = os.path.join(SAVE_DIR, filename)
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"URL: {url}\n\n{text}")

    # Recursively follow links
    for link in soup.find_all("a", href=True):
        next_url = urljoin(url, link["href"])
        if next_url.startswith("https://apisix.apache.org/docs/apisix"):
            crawl(next_url, depth - 1)

# Start the crawl
start_url = "https://apisix.apache.org/docs/apisix/getting-started/README"
crawl(start_url, depth=2)
 # Depth limits how far it follows links



