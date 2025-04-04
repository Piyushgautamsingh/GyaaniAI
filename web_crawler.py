import mechanize
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import os
import re
import tempfile
import logging
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("web_crawler")

class WebCrawler:
    def __init__(self):
        self.br = mechanize.Browser()
        self.br.set_handle_robots(False)
        self.br.addheaders = [("User-agent", "Mozilla/5.0 (X11; Linux x86_64)")]
        self.visited = set()

    def sanitize_filename(self, url: str) -> str:
        """Generate a safe filename from URL."""
        parsed = urlparse(url)
        path = parsed.path.strip("/")
        path = re.sub(r"[^\w\-_.]", "_", path)
        return path or "index"

    def crawl(self, url: str, depth: int = 1) -> Optional[str]:
        """
        Crawl a URL and return the path to the saved content.
        Returns path to the saved file or None if failed.
        """
        if depth == 0 or url in self.visited:
            return None

        logger.info(f"Crawling: {url}")
        self.visited.add(url)

        try:
            response = self.br.open(url)
            html = response.read()
        except Exception as e:
            logger.error(f"Failed to open {url}: {e}")
            return None

        soup = BeautifulSoup(html, "html.parser")
        
        # Remove unwanted elements
        for element in soup(['nav', 'footer', 'script', 'style', 'iframe', 
                           'header', 'aside', 'form', 'button', 'noscript']):
            element.decompose()
        
        # Extract clean text content
        text = soup.get_text(separator="\n", strip=True)
        
        # Create temp file
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".txt", delete=False) as f:
            f.write(f"URL: {url}\n\n{text}")
            temp_path = f.name

        # Recursively follow links (optional)
        if depth > 1:
            for link in soup.find_all("a", href=True):
                next_url = urljoin(url, link["href"])
                if self._should_follow(next_url, url):
                    self.crawl(next_url, depth - 1)

        return temp_path

    def _should_follow(self, next_url: str, base_url: str) -> bool:
        """Determine if we should follow this link."""
        parsed_next = urlparse(next_url)
        parsed_base = urlparse(base_url)
        
        # Only follow links from same domain and same path prefix
        return (parsed_next.netloc == parsed_base.netloc and 
                parsed_next.path.startswith(parsed_base.path))

def crawl_website(url: str, depth: int = 1) -> Optional[str]:
    """Convenience function to crawl a website."""
    crawler = WebCrawler()
    return crawler.crawl(url, depth)