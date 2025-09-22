import httpx
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse, urlunparse
from urllib.robotparser import RobotFileParser
from motor.motor_asyncio import AsyncIOMotorClient
from pybloom_live import BloomFilter
from playwright.async_api import async_playwright
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type, RetryError
from simhash import Simhash
import asyncio
import time
import os
from dotenv import load_dotenv
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/109.0.0.0 Safari/537.36'
HEADERS = { 'User-Agent': USER_AGENT }

class Crawler:
    def __init__(self, config):
        self.config = config
        self.base_netloc = urlparse(config['seed_url']).netloc
        self.http_client = httpx.AsyncClient(http2=False, follow_redirects=True, timeout=15, headers=HEADERS)
        self.browser = None
        
        self.robot_parsers = {}
        self.robot_lock = asyncio.Lock()

        self.http_frontier = asyncio.Queue()
        self.browser_frontier = asyncio.Queue()
        self.seen_urls = BloomFilter(capacity=1000000, error_rate=0.001)
        self.crawl_count = 0
        self.url_lock = asyncio.Lock()
        self.host_lock = asyncio.Lock()
        self.host_last_request_time = {}

        self.mongo_client = AsyncIOMotorClient(config['mongo_uri'])
        db = self.mongo_client[config['db_name']]
        self.pages_collection = db.pages1
        self.hashes_collection = db.hashes1
        self.SIMILARITY_THRESHOLD = 3

    async def get_robot_parser(self, url: str) -> RobotFileParser:
        parsed_url = urlparse(url)
        robot_url = urlunparse((parsed_url.scheme, parsed_url.netloc, '/robots.txt', '', '', ''))
        async with self.robot_lock:
            if parsed_url.netloc in self.robot_parsers:
                return self.robot_parsers[parsed_url.netloc]
            parser = RobotFileParser()
            try:
                response = await self.http_client.get(robot_url)
                if response.status_code in (200, 401, 403): parser.parse(response.text.splitlines())
                else: parser.disallow_all = True
            except Exception:
                parser.disallow_all = True
            self.robot_parsers[parsed_url.netloc] = parser
            return parser

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.ConnectError))
    )
    async def fetch_http_page(self, url: str):
        return await self.http_client.get(url)

    def needs_browser(self, html_content: str) -> bool:
        soup = BeautifulSoup(html_content, 'lxml')
        if soup.find("div", id="root") or soup.find("div", id="app"): return True
        body_text = soup.body.get_text(strip=True) if soup.body else ""
        if len(body_text) < 200 and len(soup.find_all("script", src=True)) > 1: return True
        return False

    async def process_links(self, url: str, html_content: str):
        soup = BeautifulSoup(html_content, 'lxml')
        for link_tag in soup.find_all('a', href=True):
            absolute_url = urljoin(url, link_tag['href'])
            parsed_url = urlparse(absolute_url)
            normalized_url = urlunparse((parsed_url.scheme, parsed_url.netloc, parsed_url.path, parsed_url.params, parsed_url.query, ''))
            if urlparse(normalized_url).netloc == self.base_netloc:
                async with self.url_lock:
                    if normalized_url not in self.seen_urls:
                        self.seen_urls.add(normalized_url)
                        await self.http_frontier.put(normalized_url)

    async def store_page(self, url: str, content: str):
        soup = BeautifulSoup(content, 'lxml')
        text_content = soup.body.get_text(separator=' ', strip=True) if soup.body else ""
        title = soup.title.string.strip() if soup.title else ""
        if not text_content: return

        new_hash = Simhash(text_content)
        cursor = self.hashes_collection.find({}, {"_id": 0, "hash_value": 1})
        async for existing_hash_doc in cursor:
            existing_hash = Simhash(int(existing_hash_doc['hash_value']))
            if new_hash.distance(existing_hash) < self.SIMILARITY_THRESHOLD:
                print(f"  -> [NEAR-DUPLICATE] Discarding: {url}")
                return

        page_data = {"url": url, "title": title, "text_content": text_content, "crawl_timestamp": time.time()}
        await self.pages_collection.insert_one(page_data)
        await self.hashes_collection.insert_one({"hash_value": str(new_hash.value)})
        self.crawl_count += 1
        print(f"  -> [STORED] Unique content from: {url}")

    async def http_worker(self):
        while True:
            url = await self.http_frontier.get()
            try:
                if self.config['respect_robots_txt']:
                    robot_parser = await self.get_robot_parser(url)
                    if not robot_parser.can_fetch(USER_AGENT, url):
                        print(f"  -> [ROBOTS] Disallowed: {url}")
                        continue
                
                hostname = urlparse(url).netloc
                async with self.host_lock:
                    current_time = time.time()
                    last_request = self.host_last_request_time.get(hostname, 0)
                    wait_time = max(0, self.config['politeness_delay'] - (current_time - last_request))
                    self.host_last_request_time[hostname] = current_time + wait_time
                if wait_time > 0: await asyncio.sleep(wait_time)
                
                print(f"[HTTP] Trying: {url}")
                response = await self.fetch_http_page(url)
                
                if response.status_code == 200:
                    html_content = response.text
                    if self.needs_browser(html_content):
                        await self.browser_frontier.put(url)
                    else:
                        await self.process_links(url, html_content)
                        await self.store_page(url, html_content)
                elif response.status_code == 429:
                    retry_after = response.headers.get('Retry-After')
                    wait_duration = self.config['politeness_delay'] * 10
                    if retry_after:
                        try: wait_duration = int(retry_after)
                        except ValueError:
                            try:
                                retry_date = parsedate_to_datetime(retry_after)
                                wait_duration = (retry_date - datetime.now(timezone.utc)).total_seconds()
                            except (TypeError, ValueError): pass
                    wait_duration = max(1, wait_duration)
                    print(f"  -> [RATE LIMIT] Waiting {wait_duration:.2f}s for {hostname}")
                    async with self.host_lock:
                         self.host_last_request_time[hostname] = time.time() + wait_duration
                    await self.http_frontier.put(url)
                else:
                    print(f"  -> [HTTP] Failed with status {response.status_code}: {url}")
            except Exception as e:
                error_str = str(e)
                if 'out of bounds for uint8' in error_str:
                    print(f"  -> [HTTP FALLBACK] Sending to browser: {url}")
                    await self.browser_frontier.put(url)
                elif isinstance(e, RetryError):
                     print(f"  -> [HTTP] Final fetch error after retries: {url}")
                else:
                    print(f"  -> [HTTP] General error for {url}: {e}")
            finally:
                self.http_frontier.task_done()

    async def browser_worker(self):
        while True:
            url = await self.browser_frontier.get()
            try:
                if self.config['respect_robots_txt']:
                    robot_parser = await self.get_robot_parser(url)
                    if not robot_parser.can_fetch(USER_AGENT, url):
                        print(f"  -> [ROBOTS] Disallowed: {url}")
                        continue
                print(f"[BROWSER] Rendering: {url}")
                page = await self.browser.new_page()
                await page.goto(url, wait_until='domcontentloaded', timeout=30000)
                html_content = await page.content()
                await self.process_links(url, html_content)
                await self.store_page(url, html_content)
            except Exception as e:
                print(f"  -> [BROWSER] Error rendering {url}: {e}")
            finally:
                if 'page' in locals() and page: await page.close()
                self.browser_frontier.task_done()

    async def run(self):
        async with async_playwright() as p:
            self.browser = await p.chromium.launch()
            start_time = time.time()
            parsed_seed = urlparse(self.config['seed_url'])
            normalized_seed = urlunparse((parsed_seed.scheme, parsed_seed.netloc, parsed_seed.path, parsed_seed.params, parsed_seed.query, ''))
            await self.http_frontier.put(normalized_seed)
            self.seen_urls.add(normalized_seed)
            
            workers = [asyncio.create_task(self.http_worker()) for _ in range(self.config['http_workers'])]
            workers += [asyncio.create_task(self.browser_worker()) for _ in range(self.config['browser_workers'])]

            while self.crawl_count < self.config['max_pages']:
                if self.http_frontier.empty() and self.browser_frontier.empty():
                    await asyncio.sleep(5)
                    if self.http_frontier.empty() and self.browser_frontier.empty():
                        break
                await asyncio.sleep(1)

            print("\nCrawl limit reached or frontiers are empty. Finishing in-flight tasks...")

            await self.http_frontier.join()
            await self.browser_frontier.join()

            for w in workers:
                w.cancel()
            
            await asyncio.gather(*workers, return_exceptions=True)
            
            await self.browser.close()
            await self.http_client.aclose()
            
            end_time = time.time()
            print(f"\nCrawl finished. Stored {self.crawl_count} unique pages in {end_time - start_time:.2f}s.")
        self.mongo_client.close()


async def main():
    load_dotenv()
    respect_robots_str = os.getenv("RESPECT_ROBOTS_TXT", "True").lower()
    config = {
        "seed_url": os.getenv("SEED_URL"), "mongo_uri": os.getenv("MONGO_URI"), "db_name": os.getenv("DB_NAME"),
        "http_workers": int(os.getenv("HTTP_WORKERS", "50")), "browser_workers": int(os.getenv("BROWSER_WORKERS", "5")),
        "max_pages": int(os.getenv("MAX_PAGES_TO_CRAWL", "100")), "politeness_delay": float(os.getenv("POLITENESS_DELAY_SECONDS", "1")),
        "respect_robots_txt": respect_robots_str in ('true', '1', 'yes')
    }
    if not all([config['seed_url'], config['mongo_uri'], config['db_name']]):
        print("Error: Ensure SEED_URL, MONGO_URI, and DB_NAME are in your .env file.")
        return
    crawler = Crawler(config)
    await crawler.run()

if __name__ == "__main__":
    asyncio.run(main())