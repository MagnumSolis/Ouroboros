"""
LLM Adapter - Unified interface for multiple LLM providers
Supports: Groq, Gemini, and Grok (via Puter.js for web)
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
    OPENROUTER = "openrouter"  # Free models like Llama 3.3 70B
    GEMINI = "gemini"
    PERPLEXITY = "perplexity"  # Primary for Demo (Fast & Online)
    GROK_PUTER = "grok_puter"  # Browser-only via Puter.js


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


# Rate limiting for free tier APIs
import time
_last_gemini_call = 0
GEMINI_MIN_DELAY = 2.0  # 2 seconds between calls for free tier


class GroqProvider(BaseLLMProvider):
    """Groq API provider - Ultra-fast inference"""
    
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        from groq import AsyncGroq
        self.client = AsyncGroq(api_key=api_key)
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


class GeminiProvider(BaseLLMProvider):
    """
    Google Gemini API provider using new google-genai SDK
    Docs: https://ai.google.dev/gemini-api/docs/quickstart
    """
    
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash"):
        from google import genai
        # New SDK uses Client() pattern
        self.client = genai.Client(api_key=api_key)
        self.model_name = model
    
    async def chat(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> LLMResponse:
        # Convert messages to single prompt (new SDK uses contents string)
        prompt_parts = []
        for msg in messages:
            if msg.role == "system":
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")
        
        full_prompt = "\n\n".join(prompt_parts)
        
        # Rate limiting for free tier
        global _last_gemini_call
        elapsed = time.time() - _last_gemini_call
        if elapsed < GEMINI_MIN_DELAY:
            await asyncio.sleep(GEMINI_MIN_DELAY - elapsed)
        
        # Use synchronous API wrapped in async (SDK doesn't have native async yet)
        import asyncio
        loop = asyncio.get_event_loop()
        
        def _generate():
            return self.client.models.generate_content(
                model=self.model_name,
                contents=full_prompt,
                config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens
                }
            )
        
        response = await loop.run_in_executor(None, _generate)
        _last_gemini_call = time.time()
        
        return LLMResponse(
            content=response.text,
            provider=LLMProvider.GEMINI,
            model=self.model_name,
            raw_response=response
        )
    
    async def chat_stream(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        # Convert to prompt
        prompt_parts = []
        for msg in messages:
            if msg.role == "system":
                prompt_parts.append(f"System: {msg.content}")
            elif msg.role == "user":
                prompt_parts.append(f"User: {msg.content}")
            elif msg.role == "assistant":
                prompt_parts.append(f"Assistant: {msg.content}")
        
        full_prompt = "\n\n".join(prompt_parts)
        
        # Stream using new SDK
        import asyncio
        loop = asyncio.get_event_loop()
        
        def _stream():
            return self.client.models.generate_content_stream(
                model=self.model_name,
                contents=full_prompt,
                config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens
                }
            )
        
        stream = await loop.run_in_executor(None, _stream)
        
        for chunk in stream:
            if chunk.text:
                yield chunk.text


class OpenRouterProvider(BaseLLMProvider):
    """
    OpenRouter API provider - Free access to many models
    Free models include: meta-llama/llama-3.3-70b-instruct:free
    Docs: https://openrouter.ai/docs
    """
    
    def __init__(self, api_key: str, model: str = "meta-llama/llama-3.3-70b-instruct:free"):
        import httpx
        self.api_key = api_key
        self.model = model
        self.base_url = "https://openrouter.ai/api/v1"
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def chat(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> LLMResponse:
        # Convert messages to format compatible with all OpenRouter models
        # Some models don't support 'system' role, so we prepend it to the first user message
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
            "HTTP-Referer": "https://sahayak.app",  # Required by OpenRouter
            "X-Title": "Sahayak Financial Assistant"
        }
        
        payload = {
            "model": self.model,
            "messages": formatted_messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        # Add debug logging
        logger.debug(f"OpenRouter Payload: {json.dumps(payload, default=str)[:500]}...")
        
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
    
    async def chat_stream(
        self,
        messages: List[ChatMessage],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        # For simplicity, use non-streaming for now
        response = await self.chat(messages, temperature, max_tokens, **kwargs)
        yield response.content


class PerplexityProvider(BaseLLMProvider):
    """
    Perplexity API provider - Fast, Online, and Robust
    Docs: https://docs.perplexity.ai/
    """
    
    def __init__(self, api_key: str, model: str = "llama-3.1-sonar-large-128k-online"):
        import httpx
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.perplexity.ai"
        self.client = httpx.AsyncClient(timeout=60.0)
    
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


class LLMAdapter:
    """
    Unified LLM Adapter with automatic failover
    
    Priority order (free tiers first):
    1. Perplexity - PRIMARY (Demo)
    2. Groq - FREE, 14,400 requests/day, ultra-fast
    3. OpenRouter - FREE models like Llama 3.3 70B
    4. Gemini - Has rate limits, use as backup
    """
    
    def __init__(self, preferred_provider: Optional[LLMProvider] = None):
        self.providers: Dict[LLMProvider, BaseLLMProvider] = {}
        self.preferred_provider = preferred_provider
        self._initialize_providers()
    
    def _initialize_providers(self):
        """Initialize available providers based on configured API keys"""
        
        # Perplexity - Primary for Demo (Online & Fast)
        if settings.has_perplexity:
            try:
                self.providers[LLMProvider.PERPLEXITY] = PerplexityProvider(
                    api_key=settings.perplexity_api_key,
                    model=settings.perplexity_model
                )
                logger.info(f"✅ Initialized Perplexity/Sonar (PRIMARY): {settings.perplexity_model}")
            except Exception as e:
                logger.warning(f"Failed to initialize Perplexity: {e}")
        
        # Groq - Primary free provider (14,400 requests/day)
        if settings.has_groq:
            try:
                self.providers[LLMProvider.GROQ] = GroqProvider(
                    api_key=settings.groq_api_key,
                    model=settings.groq_model
                )
                logger.info(f"✅ Initialized Groq (FREE): {settings.groq_model}")
            except Exception as e:
                logger.warning(f"Failed to initialize Groq: {e}")
        
        # OpenRouter - Free models available
        if settings.has_openrouter:
            try:
                self.providers[LLMProvider.OPENROUTER] = OpenRouterProvider(
                    api_key=settings.openrouter_api_key,
                    model=settings.openrouter_model
                )
                logger.info(f"✅ Initialized OpenRouter (FREE): {settings.openrouter_model}")
            except Exception as e:
                logger.warning(f"Failed to initialize OpenRouter: {e}")
        
        # Gemini - Backup (has strict rate limits)
        if settings.has_gemini:
            try:
                self.providers[LLMProvider.GEMINI] = GeminiProvider(
                    api_key=settings.gemini_api_key,
                    model=settings.gemini_model
                )
                logger.info(f"✅ Initialized Gemini (rate limited): {settings.gemini_model}")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini: {e}")
        
        if not self.providers:
            logger.warning("⚠️ No LLM providers configured! Get a free Groq API key at console.groq.com")
    
    def _get_provider(self, provider: Optional[LLMProvider] = None) -> BaseLLMProvider:
        """Get the best available provider"""
        
        # Use specified provider if available
        if provider and provider in self.providers:
            return self.providers[provider]
        
        # Use preferred provider if set and available
        if self.preferred_provider and self.preferred_provider in self.providers:
            return self.providers[self.preferred_provider]
        
        # Default priority: Perplexity > Groq > OpenRouter > Gemini
        for p in [LLMProvider.PERPLEXITY, LLMProvider.GROQ, LLMProvider.OPENROUTER, LLMProvider.GEMINI]:
            if p in self.providers:
                return self.providers[p]
        
        raise RuntimeError("No LLM providers available. Get free key at console.groq.com")
    
    @property
    def available_providers(self) -> List[LLMProvider]:
        """List of configured providers"""
        return list(self.providers.keys())

    async def chat(
        self,
        messages: List[ChatMessage],
        provider: Optional[LLMProvider] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> LLMResponse:
        """
        Send chat completion request with automatic failover
        """
        # If specific provider requested, try only that one
        if provider:
            llm = self._get_provider(provider)
            return await llm.chat(messages, temperature, max_tokens, **kwargs)
            
        # Otherwise, try providers in priority order
        exceptions = []
        
        # Priority: Perplexity > Groq > OpenRouter > Gemini
        priorities = [LLMProvider.PERPLEXITY, LLMProvider.GROQ, LLMProvider.OPENROUTER, LLMProvider.GEMINI]
        
        # If preferred provider exists, put it first:
        if self.preferred_provider and self.preferred_provider in priorities:
            priorities.remove(self.preferred_provider)
            priorities.insert(0, self.preferred_provider)
            
        for p in priorities:
# ...
            if p in self.providers:
                try:
                    logger.debug(f"Attempting chat with provider: {p}")
                    return await self.providers[p].chat(messages, temperature, max_tokens, **kwargs)
                except Exception as e:
                    logger.warning(f"Provider {p} failed: {e}")
                    exceptions.append(f"{p}: {e}")
                    continue
        
        # If all failed
        error_msg = "All LLM providers failed. Errors: " + "; ".join(exceptions)
        logger.error(error_msg)
        raise RuntimeError(error_msg)
    
    async def chat_stream(
        self,
        messages: List[ChatMessage],
        provider: Optional[LLMProvider] = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Stream chat completion response"""
        llm = self._get_provider(provider)
        async for chunk in llm.chat_stream(messages, temperature, max_tokens, **kwargs):
            yield chunk
    
    async def simple_chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        Simplified chat interface for quick queries
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            
        Returns:
            Response text
        """
        messages = []
        if system_prompt:
            messages.append(ChatMessage(role="system", content=system_prompt))
        messages.append(ChatMessage(role="user", content=prompt))
        
        response = await self.chat(messages, **kwargs)
        return response.content


# Puter.js integration helper for web dashboard
PUTER_JS_SCRIPT = """
// Include in HTML: <script src="https://js.puter.com/v2/"></script>

// Grok chat via Puter (browser-only)
async function grokChat(prompt, options = {}) {
    const response = await puter.ai.chat(prompt, {
        model: options.model || 'x-ai/grok-4.1-fast',
        ...options
    });
    return response.message.content;
}

// Streaming version
async function* grokStream(prompt, options = {}) {
    const response = await puter.ai.chat(prompt, {
        model: options.model || 'x-ai/grok-4.1-fast',
        stream: true,
        ...options
    });
    for await (const part of response) {
        yield part.text;
    }
}
"""
