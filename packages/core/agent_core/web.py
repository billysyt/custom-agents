"""Bounded search and public-web reader. Network input is untrusted evidence."""
import asyncio
import ipaddress
import json
import socket
from urllib.parse import urlsplit, urlunsplit
import httpx
import trafilatura

MAX_BYTES = 2_000_000

def public_url(url):
    p = urlsplit(url)
    if p.scheme not in {'http', 'https'} or not p.hostname or p.username or p.password:
        raise ValueError('Only public HTTP(S) URLs without credentials are allowed')
    if p.port not in {None, 80, 443}:
        raise ValueError('Nonstandard ports are not allowed')
    return urlunsplit((p.scheme, p.netloc, p.path or '/', p.query, ''))

async def resolve_public(url):
    url = public_url(url)
    p = urlsplit(url)
    addresses = await asyncio.get_running_loop().getaddrinfo(p.hostname, p.port or (443 if p.scheme == 'https' else 80), type=socket.SOCK_STREAM)
    if not addresses or any(not ipaddress.ip_address(a[4][0]).is_global for a in addresses):
        raise ValueError('Private, reserved and local network addresses are blocked')
    return url, addresses[0][4][0]

class Web:
    def __init__(self, search_url):
        self.search_url = search_url.rstrip('/')
        self.client = httpx.AsyncClient(timeout=25, follow_redirects=False, trust_env=False,
            headers={'User-Agent': 'CustomAgents/0.1 (+https://github.com/billysyt/custom-agents)'})
        self.reader = httpx.AsyncClient(timeout=25, follow_redirects=False, trust_env=False,
            limits=httpx.Limits(max_keepalive_connections=0), headers=dict(self.client.headers))
        # Reader egress is additionally restricted by Kubernetes NetworkPolicy.
        self.sources = []
        self.searches = []
        self.allowed_urls = set()
        self.read_attempts = 0

    async def close(self):
        await self.client.aclose()
        await self.reader.aclose()

    async def search(self, query):
        if len(self.searches) >= 4:
            return {'error': 'Search budget exhausted; read the existing results.'}
        query = query.strip()[:300]
        r = await self.client.get(self.search_url + '/search', params={
            'q': query, 'format': 'json', 'categories': 'general', 'language': 'en', 'time_range': 'month'})
        r.raise_for_status()
        data = r.json()
        results = []
        for item in data.get('results', [])[:8]:
            try:
                url = public_url(item['url'])
            except ValueError:
                continue
            self.allowed_urls.add(url)
            results.append({'title': str(item.get('title', ''))[:250], 'url': url,
                'snippet': str(item.get('content', ''))[:1000], 'published': item.get('publishedDate')})
        self.searches.append({'query': query, 'results': results})
        return results

    async def read(self, url):
        self.read_attempts += 1
        if self.read_attempts > 8:
            return {'error': 'Read budget exhausted'}
        url = public_url(url)
        if url not in self.allowed_urls:
            return {'error': 'Read a URL returned by web_search first.'}
        existing = next((s for s in self.sources if s['url'] == url), None)
        if existing:
            return existing
        original = url
        try:
            for _ in range(5):
                url, address = await resolve_public(url)
                original_url = httpx.URL(url)
                pinned_url = original_url.copy_with(host=address)
                # Connect to the validated IP while checking TLS for the original hostname.
                # Disable keepalive so a shared CDN IP cannot reuse another hostname's TLS session.
                async with self.reader.stream('GET', pinned_url, headers={'Host': original_url.netloc.decode()},
                        extensions={'sni_hostname': original_url.host}) as r:
                    if r.is_redirect:
                        from urllib.parse import urljoin
                        url = urljoin(url, r.headers['location'])
                        continue
                    r.raise_for_status()
                    if not any(t in r.headers.get('content-type', '') for t in ['text/html', 'text/plain', 'application/xhtml+xml']):
                        raise ValueError('Only HTML/text pages can be read')
                    chunks, size = [], 0
                    async for chunk in r.aiter_bytes():
                        size += len(chunk)
                        if size > MAX_BYTES:
                            raise ValueError('Page exceeds the 2 MB limit')
                        chunks.append(chunk)
                    html = b''.join(chunks).decode(r.encoding or 'utf-8', errors='replace')
                    extracted = await asyncio.to_thread(trafilatura.bare_extraction, html, url=url,
                        with_metadata=True, include_comments=False, favor_precision=True, as_dict=True)
                    if not extracted or len(extracted.get('text', '')) < 150:
                        raise ValueError('No readable article content; do not infer it from the snippet')
                    source = {'id': len(self.sources)+1, 'url': original, 'resolved_url': url,
                        'title': extracted.get('title') or urlsplit(url).hostname,
                        'date': extracted.get('date'), 'text': extracted['text'][:14000]}
                    self.sources.append(source)
                    return source
            raise ValueError('Too many redirects')
        except (httpx.HTTPError, ValueError, OSError) as exc:
            return {'url': original, 'error': type(exc).__name__ + ': source unavailable; try another result'}
