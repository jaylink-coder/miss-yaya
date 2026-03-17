"""Web crawler infrastructure for large-scale data collection.

Provides tools for crawling web data, processing CommonCrawl WARC files,
and building custom text corpora. Designed for the 7B training run
data collection track.
"""

import os
import re
import json
import time
import hashlib
import urllib.request
import urllib.parse
import urllib.robotparser
from typing import List, Optional, Dict, Any, Iterator, Set
from pathlib import Path
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass
class CrawlResult:
    """Result from crawling a single URL."""
    url: str
    text: str
    title: str = ""
    language: str = ""
    status: int = 200
    content_length: int = 0
    crawl_time: float = 0.0
    error: Optional[str] = None


class RobotsChecker:
    """Check robots.txt compliance before crawling.

    Caches robots.txt files per domain to avoid repeated fetches.
    """

    def __init__(self, user_agent: str = "YayaBot/1.0"):
        self.user_agent = user_agent
        self._cache: Dict[str, urllib.robotparser.RobotFileParser] = {}

    def _get_robots_url(self, url: str) -> str:
        parsed = urllib.parse.urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}/robots.txt"

    def can_fetch(self, url: str) -> bool:
        """Check if we're allowed to fetch this URL per robots.txt."""
        robots_url = self._get_robots_url(url)

        if robots_url not in self._cache:
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(robots_url)
            try:
                rp.read()
            except Exception:
                # If we can't read robots.txt, assume allowed
                return True
            self._cache[robots_url] = rp

        return self._cache[robots_url].can_fetch(self.user_agent, url)


class TextExtractor:
    """Extract clean text from HTML content.

    Uses a lightweight approach with regex-based tag stripping.
    For production, consider trafilatura or readability-lxml.
    """

    # Tags whose content should be completely removed
    REMOVE_TAGS = {"script", "style", "nav", "header", "footer", "aside", "noscript"}

    # Block-level tags that should add newlines
    BLOCK_TAGS = {"p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "br", "article", "section"}

    def __init__(self, min_text_length: int = 100):
        self.min_text_length = min_text_length
        self._remove_pattern = re.compile(
            r"<(" + "|".join(self.REMOVE_TAGS) + r")[^>]*>.*?</\1>",
            re.DOTALL | re.IGNORECASE,
        )
        self._tag_pattern = re.compile(r"<[^>]+>")
        self._whitespace_pattern = re.compile(r"\s+")
        self._title_pattern = re.compile(r"<title[^>]*>(.*?)</title>", re.DOTALL | re.IGNORECASE)

    def extract(self, html: str) -> Dict[str, str]:
        """Extract clean text and title from HTML.

        Args:
            html: Raw HTML string.

        Returns:
            Dict with 'text' and 'title' keys.
        """
        # Extract title
        title_match = self._title_pattern.search(html)
        title = title_match.group(1).strip() if title_match else ""
        title = self._tag_pattern.sub("", title)

        # Remove unwanted tag blocks
        text = self._remove_pattern.sub(" ", html)

        # Add newlines for block elements
        for tag in self.BLOCK_TAGS:
            text = re.sub(f"</?{tag}[^>]*>", "\n", text, flags=re.IGNORECASE)

        # Strip remaining tags
        text = self._tag_pattern.sub(" ", text)

        # Decode HTML entities
        text = text.replace("&amp;", "&")
        text = text.replace("&lt;", "<")
        text = text.replace("&gt;", ">")
        text = text.replace("&quot;", '"')
        text = text.replace("&#39;", "'")
        text = text.replace("&nbsp;", " ")

        # Clean whitespace
        text = self._whitespace_pattern.sub(" ", text)
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(line for line in lines if line)

        return {"text": text.strip(), "title": title.strip()}


class WebCrawler:
    """Simple web crawler for collecting text data.

    Features:
    - robots.txt compliance
    - Rate limiting
    - URL deduplication
    - HTML text extraction
    - JSONL output
    """

    def __init__(
        self,
        output_dir: str,
        user_agent: str = "YayaBot/1.0 (AI training data collection)",
        rate_limit: float = 1.0,
        max_workers: int = 4,
        respect_robots: bool = True,
        max_page_size: int = 5 * 1024 * 1024,
    ):
        self.output_dir = output_dir
        self.user_agent = user_agent
        self.rate_limit = rate_limit
        self.max_workers = max_workers
        self.max_page_size = max_page_size

        self.robots_checker = RobotsChecker(user_agent) if respect_robots else None
        self.extractor = TextExtractor()
        self._seen_urls: Set[str] = set()
        self._crawl_count = 0
        self._error_count = 0

        os.makedirs(output_dir, exist_ok=True)

    def _normalize_url(self, url: str) -> str:
        """Normalize URL for deduplication."""
        parsed = urllib.parse.urlparse(url)
        # Remove fragments, normalize path
        normalized = urllib.parse.urlunparse((
            parsed.scheme.lower(),
            parsed.netloc.lower(),
            parsed.path.rstrip("/"),
            parsed.params,
            parsed.query,
            "",  # Remove fragment
        ))
        return normalized

    def _fetch_url(self, url: str, timeout: int = 15) -> CrawlResult:
        """Fetch a single URL and extract text."""
        start = time.time()

        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": self.user_agent},
            )
            with urllib.request.urlopen(req, timeout=timeout) as response:
                if response.status != 200:
                    return CrawlResult(url=url, text="", status=response.status)

                content_type = response.headers.get("content-type", "")
                if "text/html" not in content_type.lower():
                    return CrawlResult(url=url, text="", status=response.status,
                                       error="Not HTML")

                html = response.read(self.max_page_size).decode("utf-8", errors="ignore")

            extracted = self.extractor.extract(html)
            elapsed = time.time() - start

            return CrawlResult(
                url=url,
                text=extracted["text"],
                title=extracted["title"],
                content_length=len(extracted["text"]),
                crawl_time=elapsed,
            )

        except Exception as e:
            return CrawlResult(
                url=url, text="", error=str(e),
                crawl_time=time.time() - start,
            )

    def crawl_url(self, url: str) -> Optional[CrawlResult]:
        """Crawl a single URL with all checks.

        Args:
            url: URL to crawl.

        Returns:
            CrawlResult or None if skipped.
        """
        url = self._normalize_url(url)

        # Dedup
        if url in self._seen_urls:
            return None
        self._seen_urls.add(url)

        # Robots check
        if self.robots_checker and not self.robots_checker.can_fetch(url):
            return None

        # Rate limiting
        time.sleep(self.rate_limit)

        result = self._fetch_url(url)
        self._crawl_count += 1
        if result.error:
            self._error_count += 1

        return result

    def crawl_urls(
        self,
        urls: List[str],
        output_file: Optional[str] = None,
        min_text_length: int = 200,
    ) -> List[CrawlResult]:
        """Crawl a list of URLs sequentially.

        Args:
            urls: List of URLs to crawl.
            output_file: Optional JSONL output file.
            min_text_length: Minimum text length to keep.

        Returns:
            List of successful CrawlResults.
        """
        if output_file is None:
            output_file = os.path.join(self.output_dir, "crawled.jsonl")

        results = []

        with open(output_file, "a", encoding="utf-8") as f:
            for i, url in enumerate(urls):
                result = self.crawl_url(url)
                if result is None:
                    continue

                if result.text and len(result.text) >= min_text_length:
                    record = {
                        "text": result.text,
                        "url": result.url,
                        "title": result.title,
                        "source": "web_crawl",
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    results.append(result)

                if (i + 1) % 100 == 0:
                    print(f"  Crawled {i+1}/{len(urls)}: {len(results)} kept, {self._error_count} errors")

        print(f"Crawl complete: {len(results)} docs from {len(urls)} URLs")
        return results

    @property
    def stats(self) -> Dict[str, Any]:
        return {
            "urls_crawled": self._crawl_count,
            "errors": self._error_count,
            "urls_seen": len(self._seen_urls),
        }


class CommonCrawlProcessor:
    """Process CommonCrawl WARC/WET files for text extraction.

    CommonCrawl provides petabytes of web data. This processor
    handles downloading and extracting text from WET files
    (pre-extracted text format).

    Usage:
        processor = CommonCrawlProcessor(output_dir="data/raw/commoncrawl")
        processor.process_wet_paths(paths_file="wet.paths", max_files=10)
    """

    WET_BASE_URL = "https://data.commoncrawl.org/"

    def __init__(
        self,
        output_dir: str,
        min_doc_length: int = 200,
        max_doc_length: int = 100000,
        language: str = "en",
    ):
        self.output_dir = output_dir
        self.min_doc_length = min_doc_length
        self.max_doc_length = max_doc_length
        self.language = language

        os.makedirs(output_dir, exist_ok=True)

    def download_wet_paths(self, crawl_id: str = "CC-MAIN-2024-10") -> str:
        """Download the WET paths file for a CommonCrawl crawl.

        Args:
            crawl_id: CommonCrawl crawl identifier.

        Returns:
            Path to downloaded paths file.
        """
        url = f"{self.WET_BASE_URL}crawl-data/{crawl_id}/wet.paths.gz"
        paths_file = os.path.join(self.output_dir, f"{crawl_id}_wet.paths.gz")

        if not os.path.exists(paths_file):
            print(f"Downloading WET paths: {url}")
            urllib.request.urlretrieve(url, paths_file)

        return paths_file

    def process_wet_file(self, wet_path: str, output_file: str) -> int:
        """Process a single WET file and extract documents.

        WET format: WARC records with pre-extracted text.
        Each record starts with WARC/1.0 header.

        Args:
            wet_path: URL or local path to WET file.
            output_file: Output JSONL path.

        Returns:
            Number of documents extracted.
        """
        import gzip

        count = 0

        # Download if URL
        if wet_path.startswith("http"):
            local_path = os.path.join(self.output_dir, "temp_wet.gz")
            urllib.request.urlretrieve(wet_path, local_path)
            wet_path = local_path

        with gzip.open(wet_path, "rt", encoding="utf-8", errors="ignore") as fin, \
             open(output_file, "a", encoding="utf-8") as fout:

            current_text = []
            current_url = ""
            in_content = False

            for line in fin:
                if line.startswith("WARC/1.0"):
                    # Save previous record
                    if current_text and current_url:
                        text = "\n".join(current_text).strip()
                        if self.min_doc_length <= len(text) <= self.max_doc_length:
                            record = {
                                "text": text,
                                "url": current_url,
                                "source": "commoncrawl",
                            }
                            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                            count += 1

                    current_text = []
                    current_url = ""
                    in_content = False

                elif line.startswith("WARC-Target-URI:"):
                    current_url = line.split(":", 1)[1].strip()

                elif line.strip() == "" and not in_content:
                    in_content = True

                elif in_content:
                    current_text.append(line.rstrip())

        return count

    def process_wet_paths(
        self,
        paths_file: str,
        max_files: int = 10,
        output_prefix: str = "cc",
    ) -> Dict[str, Any]:
        """Process multiple WET files from a paths list.

        Args:
            paths_file: File containing WET file paths (one per line).
            max_files: Maximum number of WET files to process.
            output_prefix: Prefix for output JSONL files.

        Returns:
            Processing statistics.
        """
        import gzip

        # Read paths
        opener = gzip.open if paths_file.endswith(".gz") else open
        with opener(paths_file, "rt") as f:
            paths = [line.strip() for line in f if line.strip()]

        paths = paths[:max_files]
        print(f"Processing {len(paths)} WET files")

        total_docs = 0
        for i, path in enumerate(paths):
            url = f"{self.WET_BASE_URL}{path}"
            output_file = os.path.join(self.output_dir, f"{output_prefix}_{i:05d}.jsonl")

            print(f"  [{i+1}/{len(paths)}] {path}", end="", flush=True)
            try:
                count = self.process_wet_file(url, output_file)
                total_docs += count
                print(f" -> {count:,} docs")
            except Exception as e:
                print(f" -> ERROR: {e}")

        print(f"\nTotal: {total_docs:,} documents from {len(paths)} WET files")
        return {"total_docs": total_docs, "files_processed": len(paths)}
