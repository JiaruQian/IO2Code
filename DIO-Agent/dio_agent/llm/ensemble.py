"""
Model ensemble for LLMs
"""

import asyncio
import logging
import random
from typing import Dict, List, Optional, Tuple

from dio_agent.llm.base import LLMInterface
from dio_agent.llm.openai import OpenAILLM
from dio_agent.config import LLMModelConfig

logger = logging.getLogger(__name__)


class LLMEnsemble:
    """Ensemble of LLMs"""

    def __init__(self, models_cfg: List[LLMModelConfig]):
        self.models_cfg = models_cfg
        self._last_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "llm_calls": 0,
        }

        # Initialize models from the configuration
        self.models = [
            model_cfg.init_client(model_cfg) if model_cfg.init_client else OpenAILLM(model_cfg)
            for model_cfg in models_cfg
        ]

        # Extract and normalize model weights
        self.weights = [model.weight for model in models_cfg]
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

        # Set up random state for deterministic model selection
        self.random_state = random.Random()
        # Initialize with seed from first model's config if available
        if (
            models_cfg
            and hasattr(models_cfg[0], "random_seed")
            and models_cfg[0].random_seed is not None
        ):
            self.random_state.seed(models_cfg[0].random_seed)
            logger.debug(
                f"LLMEnsemble: Set random seed to {models_cfg[0].random_seed} for deterministic model selection"
            )

        # Only log if we have multiple models or this is the first ensemble
        if len(models_cfg) > 1 or not hasattr(logger, "_ensemble_logged"):
            logger.info(
                f"Initialized LLM ensemble with models: "
                + ", ".join(
                    f"{model.name} (weight: {weight:.2f})"
                    for model, weight in zip(models_cfg, self.weights)
                )
            )
            logger._ensemble_logged = True

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using a randomly selected model based on weights"""
        model = self._sample_model()
        response = await model.generate(prompt, **kwargs)
        self._capture_usage_from_model(model)
        return response

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        """Generate text using a system message and conversational context"""
        model = self._sample_model()
        response = await model.generate_with_context(system_message, messages, **kwargs)
        self._capture_usage_from_model(model)
        return response

    def _sample_model(self) -> LLMInterface:
        """Sample a model from the ensemble based on weights"""
        index = self.random_state.choices(range(len(self.models)), weights=self.weights, k=1)[0]
        sampled_model = self.models[index]
        logger.info(f"Sampled model: {vars(sampled_model)['model']}")
        return sampled_model

    def _capture_usage_from_model(self, model: LLMInterface) -> None:
        if not hasattr(model, "consume_last_usage"):
            self._last_usage = {
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
                "llm_calls": 0,
            }
            return
        usage = model.consume_last_usage()
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        total_tokens = int(usage.get("total_tokens", 0) or 0)
        if total_tokens <= 0:
            total_tokens = prompt_tokens + completion_tokens
        self._last_usage = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
            "llm_calls": 1,
        }

    def consume_last_usage(self) -> Dict[str, int]:
        usage = self._last_usage or {}
        consumed = {
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
            "llm_calls": int(usage.get("llm_calls", 0) or 0),
        }
        self._last_usage = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "llm_calls": 0,
        }
        return consumed

    async def generate_multiple(self, prompt: str, n: int, **kwargs) -> List[str]:
        """Generate multiple texts in parallel"""
        tasks = [self.generate(prompt, **kwargs) for _ in range(n)]
        return await asyncio.gather(*tasks)

    async def parallel_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for multiple prompts in parallel"""
        tasks = [self.generate(prompt, **kwargs) for prompt in prompts]
        return await asyncio.gather(*tasks)

    async def generate_all_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        """Generate text using a all available models and average their returned metrics"""
        responses = []
        for model in self.models:
            responses.append(await model.generate_with_context(system_message, messages, **kwargs))
        return responses
