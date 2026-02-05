"""
Web Search Tool - Perplexity/Google integration for real-time information

For grounding agent responses with current information:
- Latest RBI circulars and guidelines
- Real-time scheme updates
- News about financial fraud patterns
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass

from loguru import logger


@dataclass
class SearchResult:
    """Web search result"""
    title: str
    url: str
    snippet: str
    relevance: float = 0.0


class WebSearchTool:
    """
    Web search integration for real-time grounding
    
    Can use:
    - Perplexity API (if available)
    - DuckDuckGo (free, no API)
    - Serper.dev (Google search API)
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize web search tool
        
        Args:
            api_key: Optional API key for Perplexity/Serper
        """
        self.api_key = api_key
        self._ddg_available = False
        
        # Try to import DuckDuckGo as fallback
        try:
            from duckduckgo_search import DDGS
            self._ddg = DDGS()
            self._ddg_available = True
            logger.info("WebSearchTool initialized with DuckDuckGo")
        except ImportError:
            self._ddg = None
            logger.warning("DuckDuckGo not available, install duckduckgo-search for free web search")
    
    async def search(
        self,
        query: str,
        num_results: int = 5,
        language: str = "en"
    ) -> List[SearchResult]:
        """
        Search the web for information
        
        Args:
            query: Search query
            num_results: Number of results to return
            language: Language preference (en, hi)
            
        Returns:
            List of search results
        """
        if self._ddg_available:
            return await self._search_ddg(query, num_results)
        else:
            logger.warning("No search provider available")
            return []
    
    async def _search_ddg(
        self,
        query: str,
        num_results: int
    ) -> List[SearchResult]:
        """Search using DuckDuckGo"""
        try:
            results = []
            for r in self._ddg.text(query, max_results=num_results):
                results.append(SearchResult(
                    title=r.get("title", ""),
                    url=r.get("href", ""),
                    snippet=r.get("body", ""),
                    relevance=0.8
                ))
            return results
        except Exception as e:
            logger.error(f"DuckDuckGo search error: {e}")
            return []
    
    async def search_rbi_updates(
        self,
        topic: str,
        num_results: int = 3
    ) -> List[SearchResult]:
        """Search specifically for RBI guidelines and circulars"""
        query = f"RBI {topic} site:rbi.org.in"
        return await self.search(query, num_results)
    
    async def search_schemes(
        self,
        scheme_name: str,
        num_results: int = 3
    ) -> List[SearchResult]:
        """Search for government scheme information"""
        query = f"{scheme_name} eligibility benefits site:india.gov.in OR site:pmjdy.gov.in"
        return await self.search(query, num_results)
    
    async def search_fraud_news(
        self,
        fraud_type: str,
        num_results: int = 5
    ) -> List[SearchResult]:
        """Search for recent fraud news and warnings"""
        query = f"{fraud_type} fraud scam India warning 2024"
        return await self.search(query, num_results)
