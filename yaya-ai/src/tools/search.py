"""Search tool — looks up information from Yaya's knowledge base or Wikipedia."""

import json
import os
import urllib.request
import urllib.parse
from .base import BaseTool, ToolResult


class SearchTool(BaseTool):
    name = 'search'
    description = 'Searches for information on a topic. Input: a search query.'

    def run(self, input_text: str) -> ToolResult:
        query = input_text.strip()
        try:
            # Use Wikipedia's public API — no key needed
            params = urllib.parse.urlencode({
                'action': 'query',
                'format': 'json',
                'titles': query,
                'prop': 'extracts',
                'exintro': True,
                'explaintext': True,
                'redirects': 1,
                'exsentences': 4,
            })
            url = f'https://en.wikipedia.org/w/api.php?{params}'
            req = urllib.request.Request(url, headers={'User-Agent': 'YayaAI/1.0'})

            with urllib.request.urlopen(req, timeout=5) as resp:
                data = json.loads(resp.read().decode('utf-8'))

            pages = data.get('query', {}).get('pages', {})
            for page in pages.values():
                if 'missing' in page:
                    return ToolResult(
                        tool_name=self.name,
                        success=False,
                        output='',
                        error=f'No Wikipedia article found for: {query}',
                    )
                extract = page.get('extract', '').strip()
                if extract:
                    # Trim to reasonable length
                    sentences = extract.split('. ')[:4]
                    summary = '. '.join(sentences).strip()
                    if not summary.endswith('.'):
                        summary += '.'
                    return ToolResult(
                        tool_name=self.name,
                        success=True,
                        output=f'[Wikipedia: {page["title"]}]\n{summary}',
                    )

            return ToolResult(
                tool_name=self.name,
                success=False,
                output='',
                error=f'No results found for: {query}',
            )

        except Exception as e:
            return ToolResult(
                tool_name=self.name,
                success=False,
                output='',
                error=f'Search failed: {e}',
            )
