"""
LLM Adapter - Unified interface for multiple LLM providers
Supports: Groq and Perplexity (Two-pronged approach)
"""

from typing import Optional, List, Dict, Any, AsyncGenerator
from abc import ABC, abstractmethod
from enum import Enum
from dataclasses import dataclass
import json

from loguru import logger

from ..config import settings


class LLMProvider(str, Enum):
    """Available LLM providers"""
    GROQ = "groq"
    PERPLEXITY = "perplexity"  # Primary for Online Search / Knowledge
    OPENROUTER = "openrouter"  # Fallback


@dataclass
class ChatMessage:
    """Standard chat message format"""
    role: str  # system, user, assistant
    content: str


@dataclass
class LLMResponse:
    """Standard LLM response format"""
    content: str
    provider: LLMProvider
    model: str
    usage: Optional[Dict[str, int]] = None
    raw_response: Optional[Any] = None


class BaseLLMProvider(ABC):
    """Base class for LLM providers"""
    
    @abstractmethod
    async def chat(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> LLMResponse:
        pass
    
    @abstractmethod
    async def chat_stream(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion response"""
        pass


class GroqProvider(BaseLLMProvider):
    """Groq API provider - Ultra-fast inference for Reasoning & Tool Use"""
    
    
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        from groq import AsyncGroq
        # Increase timeout to 120s for stability during demo
        self.client = AsyncGroq(api_key=api_key, timeout=120.0)
        self.model = model
    
    async def chat(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> LLMResponse:
        formatted_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        response = await self.client.chat.completions.create(
            model=self.model,
            messages=formatted_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        return LLMResponse(
            content=response.choices[0].message.content,
            provider=LLMProvider.GROQ,
            model=self.model,
            usage={
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            },
            raw_response=response
        )
    
    async def chat_stream(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        formatted_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=formatted_messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content


class PerplexityProvider(BaseLLMProvider):
    """
    Perplexity API provider - Fast, Online, and Robust
    """
    
    def __init__(self, api_key: str, model: str = "llama-3.1-sonar-large-128k-online"):
        import httpx
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.perplexity.ai"
        # Increase timeout to 120s
        self.client = httpx.AsyncClient(timeout=120.0)
    
    async def chat(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.2, # Lower temp for factual queries
        max_tokens: int = 1024,
        **kwargs
    ) -> LLMResponse:
        formatted_messages = [
            {"role": msg.role, "content": msg.content}
            for msg in messages
        ]
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "return_citations": True
        }
        
        response = await self.client.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code >= 400:
            logger.error(f"Perplexity Error {response.status_code}: {response.text}")
            
        response.raise_for_status()
        data = response.json()
        
        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            provider=LLMProvider.PERPLEXITY,
            model=self.model,
            usage=data.get("usage"),
            raw_response=data
        )
    
    async def chat_stream(self, messages, **kwargs):
        # Non-streaming for stability first
        response = await self.chat(messages, **kwargs)
        yield response.content


class OpenRouterProvider(BaseLLMProvider):
    """
    OpenRouter API provider - Free fallback
    """
    
    def __init__(self, api_key: str, model: str = "meta-llama/llama-3.3-70b-instruct:free"):
        import httpx
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1"
        # Increase timeout to 120s
        self.client = httpx.AsyncClient(timeout=120.0)
    
    async def chat(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> LLMResponse:
        formatted_messages = []
        system_content = ""
        
        for msg in messages:
            if msg.role == "system":
                system_content += f"{msg.content}\n\n"
            else:
                content = msg.content
                if system_content and msg.role == "user":
                    content = f"System Instructions:\n{system_content}\nUser Query:\n{msg.content}"
                    system_content = ""  # Clear after appending
                
                formatted_messages.append({"role": msg.role, "content": content})
                
        # If system prompt remains (no user message), add it as user message
        if system_content:
             formatted_messages.append({"role": "user", "content": f"System Instructions:\n{system_content}"})
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://sahayak.app",
            "X-Title": "Sahayak Financial Assistant"
        }
        
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = await self.client.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code >= 400:
            logger.error(f"OpenRouter Error {response.status_code}: {response.text}")
            
        response.raise_for_status()
        data = response.json()
        
        return LLMResponse(
            content=data["choices"][0]["message"]["content"],
            provider=LLMProvider.OPENROUTER,
            model=self.model,
            usage=data.get("usage"),
            raw_response=data
        )
    
    async def chat_stream(self, messages, **kwargs):
        response = await self.chat(messages, **kwargs)
        yield response.content


class LLMAdapter:
    """
    Unified LLM Adapter: Groq (Reasoning) + Perplexity (Knowledge)
    """
    
    def __init__(self, preferred_provider: Optional[LLMProvider] = None):
        self.providers: Dict[LLMProvider, BaseLLMProvider] = {}
        self.preferred_provider = preferred_provider
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize available providers"""
        
        # 1. Perplexity (Knowledge/Search)
        if settings.has_perplexity:
            try:
                self.providers[LLMProvider.PERPLEXITY] = PerplexityProvider(
                    api_key=settings.perplexity_api_key,
                    model=settings.perplexity_model
                )
                logger.info(f"✅ Initialized Perplexity: {settings.perplexity_model}")
            except Exception as e:
                logger.warning(f"Failed to initialize Perplexity: {e}")
        
        # 2. Groq (Reasoning/Speed)
        if settings.has_groq:
            try:
                self.providers[LLMProvider.GROQ] = GroqProvider(
                    api_key=settings.groq_api_key,
                    model=settings.groq_model
                )
                logger.info(f"✅ Initialized Groq: {settings.groq_model}")
            except Exception as e:
                logger.warning(f"Failed to initialize Groq: {e}")
        
        # 3. OpenRouter (Fallback)
        if settings.has_openrouter:
            try:
                self.providers[LLMProvider.OPENROUTER] = OpenRouterProvider(
                    api_key=settings.openrouter_api_key,
                    model=settings.openrouter_model
                )
                logger.info(f"✅ Initialized OpenRouter: {settings.openrouter_model}")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenRouter: {e}")
        
        if not self.providers:
            logger.error("⚠️ No LLM providers configured! Please set GROQ_API_KEY or PERPLEXITY_API_KEY.")
    
    def _get_provider(self, provider: Optional[LLMProvider] = None) -> BaseLLMProvider:
        """Get the requested or best available provider"""
        
        # Use specified provider if available
        if provider and provider in self.providers:
            return self.providers[provider]
        
        # Use preferred provider if set
        if self.preferred_provider and self.preferred_provider in self.providers:
            return self.providers[self.preferred_provider]
        
        # Default priority: Groq > Perplexity > OpenRouter
        for p in [LLMProvider.GROQ, LLMProvider.PERPLEXITY, LLMProvider.OPENROUTER]:
            if p in self.providers:
                return self.providers[p]
        
        raise RuntimeError("No LLM providers available.")
    
    async def chat(
        self,
        messages: List[ChatMessage],
        provider: Optional[LLMProvider] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> LLMResponse:
        """
        Send chat completion request
        """
        try:
            llm = self._get_provider(provider)
            return await llm.chat(messages, temperature, max_tokens, **kwargs)
        except Exception as e:
            logger.error(f"LLM Chat Error: {e}")
            raise

    async def chat_stream(
        self,
        messages: List[ChatMessage],
        provider: Optional[LLMProvider] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion"""
        try:
            llm = self._get_provider(provider)
            async for chunk in llm.chat_stream(messages, temperature, max_tokens, **kwargs):
                yield chunk
        except Exception as e:
            logger.error(f"LLM Stream Error: {e}")
            raise
    
    async def simple_chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Simplified chat interface"""
        messages = []
        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))
        messages.append(ChatMessage(role="user", content=prompt))
        
        response = await self.chat(messages, **kwargs)
        return response.content

