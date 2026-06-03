"""
Apply environment-variable overrides to compiled model registry entries.

YAML model_registry.yaml holds defaults; this module overlays deployment/model_id
values from process env at registry build time. Optional GPT-5.x Azure aliases
may resolve to an empty deployment when their env var is unset — route validation
must reject active routes that reference them without a configured deployment.
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from schemas.llm_routing import ModelConfig

logger = logging.getLogger(__name__)

# alias → (env_var_name, yaml_default_when_env_blank)
_AZURE_DEPLOYMENT_ENV: dict[str, tuple[str, str]] = {
    "openai_gpt_4_1": ("AZURE_OPENAI_DEPLOYMENT_GPT_4_1", "gpt-4.1"),
    "openai_gpt_4_1_mini": ("AZURE_OPENAI_DEPLOYMENT_GPT_4_1_MINI", "gpt-4.1-mini"),
    "openai_o4_mini": ("AZURE_OPENAI_DEPLOYMENT_O4_MINI", "o4-mini"),
    "openai_o3": ("AZURE_OPENAI_DEPLOYMENT_O3", "o3"),
    "openai_gpt_5_4": ("AZURE_OPENAI_DEPLOYMENT_GPT_5_4", ""),
    "openai_gpt_5_4_mini": ("AZURE_OPENAI_DEPLOYMENT_GPT_5_4_MINI", ""),
    "openai_gpt_5_5": ("AZURE_OPENAI_DEPLOYMENT_GPT_5_5", ""),
    "doubt_solver_classifier": ("AZURE_OPENAI_DEPLOYMENT_GPT_4_1_MINI", "gpt-4.1-mini"),
    "doubt_solver_classifier_strong": ("AZURE_OPENAI_DEPLOYMENT_GPT_4_1", "gpt-4.1"),
    "math_basic_generator": ("AZURE_OPENAI_DEPLOYMENT_GPT_4_1_MINI", "gpt-4.1-mini"),
    "math_intermediate_generator": ("AZURE_OPENAI_DEPLOYMENT_GPT_4_1_MINI", "gpt-4.1-mini"),
    "reasoning_intermediate_generator": ("AZURE_OPENAI_DEPLOYMENT_O4_MINI", "o4-mini"),
    "reasoning_advanced_generator": ("AZURE_OPENAI_DEPLOYMENT_O4_MINI", "o4-mini"),
    "math_cat_advanced_generator": ("AZURE_OPENAI_DEPLOYMENT_O3", "o3"),
    "reasoning_sbi_po_complex_generator": ("AZURE_OPENAI_DEPLOYMENT_O3", "o3"),
    "reasoning_cat_lrdi_generator": ("AZURE_OPENAI_DEPLOYMENT_O3", "o3"),
}

_GEMINI_MODEL_ENV: dict[str, tuple[str, str]] = {
    "gemini_flash_lite_text": ("GEMINI_DEFAULT_MODEL", "gemini-2.5-flash-lite"),
    "gemini_flash_text": ("GEMINI_TEXT_MODEL", "gemini-2.5-flash"),
    "gemini_image_extractor": ("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-lite"),
}

_GEMINI_TIMEOUT_ALIASES: frozenset[str] = frozenset(_GEMINI_MODEL_ENV.keys())

_DEEPSEEK_MODEL_ENV: dict[str, tuple[str, str]] = {
    "deepseek_standard_generator": ("DEEPSEEK_DEFAULT_MODEL", "deepseek-chat"),
    "deepseek_reasoning_generator": ("DEEPSEEK_REASONER_MODEL", "deepseek-reasoner"),
    "deepseek_advanced_generator": ("DEEPSEEK_ADVANCED_MODEL", "deepseek-reasoner"),
    "deepseek_v4pro": ("DEEPSEEK_V4PRO_MODEL", "deepseek-reasoner"),
    "math_advanced_generator": ("DEEPSEEK_V4PRO_MODEL", "deepseek-reasoner"),
}

_DEEPSEEK_TIMEOUT_ALIASES: dict[str, int] = {
    "deepseek_standard_generator": 60,
    "deepseek_reasoning_generator": 90,
    "deepseek_advanced_generator": 90,
}

_AZURE_DEEPSEEK_DEPLOYMENT_ENV: dict[str, tuple[str, str]] = {
    "deepseek_azure_reasoning_generator": ("AZURE_DEEPSEEK_REASONER_DEPLOYMENT", ""),
    "deepseek_azure_standard_generator": ("AZURE_DEEPSEEK_CHAT_DEPLOYMENT", ""),
    "deepseek_azure_advanced_generator": ("AZURE_DEEPSEEK_ADVANCED_DEPLOYMENT", ""),
}

_AZURE_DEEPSEEK_TIMEOUT_ALIASES: dict[str, int] = {
    "deepseek_azure_reasoning_generator": 90,
    "deepseek_azure_standard_generator": 60,
    "deepseek_azure_advanced_generator": 90,
}


def _env_or_default(env_var: str, default: str) -> str:
    value = os.getenv(env_var, "").strip()
    return value if value else default


def apply_model_registry_env_overrides(model_map: dict[str, ModelConfig]) -> None:
    """Mutate *model_map* in place with env-based deployment/model overrides."""
    gemini_timeout = int(os.getenv("GEMINI_TIMEOUT_SECONDS", "30"))
    deepseek_timeout = int(os.getenv("DEEPSEEK_TIMEOUT_SECONDS", "60"))
    azure_deepseek_timeout = int(os.getenv("AZURE_DEEPSEEK_TIMEOUT_SECONDS", "90"))

    for alias, cfg in list(model_map.items()):
        updated: ModelConfig | None = None

        if alias in _AZURE_DEPLOYMENT_ENV and cfg.provider == "azure_openai":
            env_var, default = _AZURE_DEPLOYMENT_ENV[alias]
            deployment = _env_or_default(env_var, default)
            if deployment != (cfg.deployment or ""):
                updated = cfg.model_copy(update={"deployment": deployment})

        if alias in _GEMINI_MODEL_ENV and cfg.provider == "gemini":
            env_var, default = _GEMINI_MODEL_ENV[alias]
            model_id = _env_or_default(env_var, default)
            timeout = gemini_timeout if alias in _GEMINI_TIMEOUT_ALIASES else cfg.timeout_seconds
            base = updated or cfg
            if model_id != (base.model_id or "") or timeout != base.timeout_seconds:
                updated = base.model_copy(
                    update={"model_id": model_id, "timeout_seconds": timeout}
                )

        if alias in _DEEPSEEK_MODEL_ENV and cfg.provider == "deepseek":
            env_var, default = _DEEPSEEK_MODEL_ENV[alias]
            model_id = _env_or_default(env_var, default)
            timeout = _DEEPSEEK_TIMEOUT_ALIASES.get(alias, deepseek_timeout)
            base = updated or cfg
            if model_id != (base.model_id or "") or timeout != base.timeout_seconds:
                updated = base.model_copy(
                    update={"model_id": model_id, "timeout_seconds": timeout}
                )

        if alias in _AZURE_DEEPSEEK_DEPLOYMENT_ENV and cfg.provider == "azure_openai":
            env_var, default = _AZURE_DEEPSEEK_DEPLOYMENT_ENV[alias]
            deployment = _env_or_default(env_var, default)
            timeout = _AZURE_DEEPSEEK_TIMEOUT_ALIASES.get(alias, azure_deepseek_timeout)
            base = updated or cfg
            if deployment != (base.deployment or "") or timeout != base.timeout_seconds:
                updated = base.model_copy(
                    update={"deployment": deployment, "timeout_seconds": timeout}
                )

        if updated is not None:
            model_map[alias] = updated

    logger.debug(
        "model_registry_env  applied  azure_overrides=%d  gemini=%d  deepseek=%d"
        "  azure_deepseek=%d",
        len(_AZURE_DEPLOYMENT_ENV),
        len(_GEMINI_MODEL_ENV),
        len(_DEEPSEEK_MODEL_ENV),
        len(_AZURE_DEEPSEEK_DEPLOYMENT_ENV),
    )
