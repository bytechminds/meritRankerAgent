"""
app/config.py
-------------
Application settings loaded from environment variables.

Priority order (highest → lowest):
  1. Real environment variables (set by agentcore dev or shell)
  2. app/.env.local  (local secrets, gitignored)
  3. Hardcoded defaults below

Never log secrets. Keep this module import-safe and side-effect-free
except for the load_dotenv call at module load time.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load .env.local sitting next to this file.
# override=False means real env vars always win — safe for production too.
_env_file = Path(__file__).parent / ".env.local"
load_dotenv(_env_file, override=False)


class ConfigurationError(Exception):
    """Raised when the application is misconfigured at startup.

    This is a fail-fast error — it is raised during module-level graph
    construction (before any request is processed) so operators can identify
    and fix the misconfiguration before the process accepts traffic.

    Example: ENABLE_ORCHESTRATED_DOUBT_SOLVER=true with ENABLE_REAL_LLM=false
             when APP_ENV=production (would silently return mock answers).
    """

@dataclass(frozen=True)
class Settings:
    """Immutable settings snapshot.  All values come from os.environ."""

    app_env: str
    log_level: str
    model_provider: str
    # LLM routing
    enable_real_llm: bool
    llm_default_provider: str
    llm_role_config_json: str  # raw JSON string — parsed by get_llm_role_config()
    # Orchestrated doubt solver path (enabled via ENABLE_ORCHESTRATED_DOUBT_SOLVER)
    enable_orchestrated_doubt_solver: bool
    # Safety override: allow mock orchestrated executor outside normal non-production
    # gating.  Set ENABLE_ORCHESTRATED_MOCK_LLM=true ONLY for controlled internal
    # testing.  Must NEVER be set in normal production deployments.
    enable_orchestrated_mock_llm: bool
    # Bedrock Knowledge Base retrieval
    enable_kb_retrieval: bool
    bedrock_kb_id: str
    bedrock_kb_region: str  # empty string means "use AWS_REGION or boto3 default"
    bedrock_kb_max_results: int
    bedrock_kb_min_score: float | None  # None means "no minimum threshold"
    # DynamoDB record fetch
    enable_dynamodb_fetch: bool
    dynamodb_question_table: str
    dynamodb_pattern_table: str
    dynamodb_default_index: str  # empty string means "no default index"
    dynamodb_region: str  # empty string means "use AWS_REGION or boto3 default"
    # Context builder
    doubt_solver_max_context_chars: int  # hard cap on context string passed to answer generator
    # Context retrieval (Part 13.1)
    context_retrieval_timeout_ms: int
    context_max_chars: int
    context_kb_top_k: int
    context_rerank_top_n: int
    context_retrieval_version: str
    context_kb_schema_version: str  # empty string disables schemaVersion filter
    context_kb_schema_version_mandatory: bool  # when true, BROAD lane keeps schemaVersion
    context_kb_taxonomy_approved_only: bool
    classifier_confidence_fallback_threshold: float
    context_topic_hint_confidence_threshold: float
    context_max_retrieval_tags: int
    # Web search (conditional fresh context)
    web_search_enabled: bool
    web_search_provider: str
    tavily_api_key: str
    web_search_timeout_seconds: float
    web_search_max_results: int
    web_search_max_context_chars: int
    web_search_max_selected_results: int
    web_search_rerank_min_score: float
    web_search_source_strictness: str
    web_search_allow_generic_fallback: bool
    web_search_allow_exam_prep_fallback: bool
    web_search_exam_prep_max_selected_results: int
    web_search_require_official_for_exam_updates: bool
    web_search_require_trusted_for_current_affairs: bool
    web_search_min_trusted_results: int
    web_search_default_recent_days: int
    web_search_search_depth: str
    web_search_enable_extract: bool
    web_search_extract_max_urls: int
    web_search_extract_depth: str
    # Legacy optional env overrides (YAML source packs are primary)
    web_search_allowed_domains: list[str]
    web_search_blocked_domains: list[str]
    web_search_trusted_domains: list[str]
    # Answer completion / continuation
    answer_completion_marker: str
    answer_continuation_enabled: bool
    answer_continuation_max_attempts: int
    # Generator answer quality validation / rewrite
    answer_quality_validation_enabled: bool
    answer_quality_rewrite_enabled: bool
    answer_quality_math_intermediate_max_chars: int
    answer_quality_max_rewrite_attempts: int
    answer_quality_max_visible_steps: int
    answer_quality_max_display_math_blocks: int
    answer_quality_max_math_line_chars: int
    # Optional Gemini provider (not required at startup)
    gemini_api_key: str
    gemini_base_url: str
    gemini_timeout_seconds: int
    gemini_default_model: str
    gemini_image_model: str
    gemini_text_model: str
    # Optional DeepSeek provider (not required at startup)
    deepseek_api_key: str
    deepseek_base_url: str
    deepseek_timeout_seconds: int
    deepseek_default_model: str
    deepseek_reasoner_model: str
    # Optional route feature flags (YAML test routes remain inactive unless referenced)
    llm_enable_gemini_routes: bool
    llm_enable_deepseek_routes: bool
    # Azure deployment names (must match Azure portal — not public model names)
    azure_openai_deployment_gpt_4_1: str
    azure_openai_deployment_gpt_4_1_mini: str
    azure_openai_deployment_gpt_5_4: str
    azure_openai_deployment_gpt_5_4_mini: str
    azure_openai_deployment_gpt_5_5: str
    azure_openai_deployment_o4_mini: str
    azure_openai_deployment_o3: str
    azure_openai_send_reasoning_effort: bool
    deepseek_advanced_model: str
    deepseek_v4pro_model: str
    # Optional Azure-hosted DeepSeek (inactive unless route selects azure_deepseek alias)
    azure_deepseek_api_key: str
    azure_deepseek_endpoint: str
    azure_deepseek_api_version: str
    azure_deepseek_timeout_seconds: int
    azure_deepseek_reasoner_deployment: str
    azure_deepseek_chat_deployment: str
    azure_deepseek_advanced_deployment: str


# Module-level singleton — built once on first call to get_settings().
_settings: Settings | None = None


def _parse_optional_float(value: str) -> float | None:
    """Return a float if *value* is non-empty and parseable, else None."""
    stripped = value.strip()
    if not stripped:
        return None
    try:
        return float(stripped)
    except ValueError:
        return None


def _parse_confidence_threshold(value: str, *, default: float = 0.93) -> float:
    """Parse a 0.0–1.0 confidence threshold from env; fail fast on invalid values."""
    stripped = value.strip()
    if not stripped:
        return default
    try:
        parsed = float(stripped)
    except ValueError as exc:
        raise ValueError(
            "DOUBT_SOLVER_CLASSIFIER_CONFIDENCE_THRESHOLD must be a float "
            "between 0.0 and 1.0"
        ) from exc
    if not 0.0 <= parsed <= 1.0:
        raise ValueError(
            "DOUBT_SOLVER_CLASSIFIER_CONFIDENCE_THRESHOLD must be between 0.0 and 1.0"
        )
    return parsed


def _parse_domain_list(value: str) -> list[str]:
    """Parse comma-separated domain allow/block lists."""
    stripped = value.strip()
    if not stripped:
        return []
    return [part.strip() for part in stripped.split(",") if part.strip()]


def _classifier_confidence_threshold_from_env() -> float:
    """Primary classifier threshold for strong-model escalation."""
    primary = os.getenv("DOUBT_SOLVER_CLASSIFIER_CONFIDENCE_THRESHOLD", "").strip()
    if primary:
        return _parse_confidence_threshold(primary)
    legacy = os.getenv("CLASSIFIER_CONFIDENCE_FALLBACK_THRESHOLD", "").strip()
    return _parse_confidence_threshold(legacy)


def get_settings() -> Settings:
    """Return the singleton Settings instance.

    Reads os.environ at first call.  Subsequent calls return the cached object
    so environment variables are not re-read mid-request.
    """
    global _settings
    if _settings is None:
        _settings = Settings(
            app_env=os.getenv("APP_ENV", "local"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            model_provider=os.getenv("MODEL_PROVIDER", "mock"),
            enable_real_llm=os.getenv("ENABLE_REAL_LLM", "false").lower() == "true",
            llm_default_provider=os.getenv("LLM_DEFAULT_PROVIDER", "mock"),
            llm_role_config_json=os.getenv("LLM_ROLE_CONFIG_JSON", "{}"),
            enable_orchestrated_doubt_solver=(
                os.getenv("ENABLE_ORCHESTRATED_DOUBT_SOLVER", "false").lower() == "true"
            ),
            enable_orchestrated_mock_llm=(
                os.getenv("ENABLE_ORCHESTRATED_MOCK_LLM", "false").lower() == "true"
            ),
            enable_kb_retrieval=os.getenv("ENABLE_KB_RETRIEVAL", "false").lower() == "true",
            bedrock_kb_id=os.getenv("BEDROCK_KB_ID", ""),
            bedrock_kb_region=os.getenv(
                "BEDROCK_KB_REGION", os.getenv("AWS_REGION", "")
            ),
            bedrock_kb_max_results=int(os.getenv("BEDROCK_KB_MAX_RESULTS", "5")),
            bedrock_kb_min_score=_parse_optional_float(
                os.getenv("BEDROCK_KB_MIN_SCORE", "")
            ),
            enable_dynamodb_fetch=os.getenv("ENABLE_DYNAMODB_FETCH", "false").lower() == "true",
            dynamodb_question_table=os.getenv("DYNAMODB_QUESTION_TABLE", ""),
            dynamodb_pattern_table=os.getenv("DYNAMODB_PATTERN_TABLE", ""),
            dynamodb_default_index=os.getenv("DYNAMODB_DEFAULT_INDEX", ""),
            dynamodb_region=os.getenv(
                "DYNAMODB_REGION", os.getenv("AWS_REGION", "")
            ),
            doubt_solver_max_context_chars=int(
                os.getenv("DOUBT_SOLVER_MAX_CONTEXT_CHARS", "6000")
            ),
            context_retrieval_timeout_ms=int(
                os.getenv("CONTEXT_RETRIEVAL_TIMEOUT_MS", "1200")
            ),
            context_max_chars=int(os.getenv("CONTEXT_MAX_CHARS", "2500")),
            context_kb_top_k=int(os.getenv("CONTEXT_KB_TOP_K", "5")),
            context_rerank_top_n=int(os.getenv("CONTEXT_RERANK_TOP_N", "2")),
            context_retrieval_version=os.getenv("CONTEXT_RETRIEVAL_VERSION", "v1"),
            context_kb_schema_version=os.getenv("CONTEXT_KB_SCHEMA_VERSION", "v2"),
            context_kb_schema_version_mandatory=(
                os.getenv("CONTEXT_KB_SCHEMA_VERSION_MANDATORY", "false").lower() == "true"
            ),
            context_kb_taxonomy_approved_only=(
                os.getenv("CONTEXT_KB_TAXONOMY_APPROVED_ONLY", "true").lower() == "true"
            ),
            classifier_confidence_fallback_threshold=_classifier_confidence_threshold_from_env(),
            context_topic_hint_confidence_threshold=_parse_confidence_threshold(
                os.getenv("CONTEXT_TOPIC_HINT_CONFIDENCE_THRESHOLD", ""),
                default=0.85,
            ),
            context_max_retrieval_tags=int(os.getenv("CONTEXT_MAX_RETRIEVAL_TAGS", "10")),
            web_search_enabled=os.getenv("WEB_SEARCH_ENABLED", "false").lower() == "true",
            web_search_provider=os.getenv("WEB_SEARCH_PROVIDER", "tavily"),
            tavily_api_key=os.getenv("TAVILY_API_KEY", ""),
            web_search_timeout_seconds=float(os.getenv("WEB_SEARCH_TIMEOUT_SECONDS", "8")),
            web_search_max_results=int(os.getenv("WEB_SEARCH_MAX_RESULTS", "5")),
            web_search_max_context_chars=int(os.getenv("WEB_SEARCH_MAX_CONTEXT_CHARS", "2500")),
            web_search_max_selected_results=int(
                os.getenv("WEB_SEARCH_MAX_SELECTED_RESULTS", "3")
            ),
            web_search_rerank_min_score=float(
                os.getenv("WEB_SEARCH_RERANK_MIN_SCORE", "0.65")
            ),
            web_search_source_strictness=os.getenv(
                "WEB_SEARCH_SOURCE_STRICTNESS", "authoritative_first"
            ),
            web_search_allow_generic_fallback=(
                os.getenv("WEB_SEARCH_ALLOW_GENERIC_FALLBACK", "false").lower() == "true"
            ),
            web_search_allow_exam_prep_fallback=(
                os.getenv("WEB_SEARCH_ALLOW_EXAM_PREP_FALLBACK", "true").lower() == "true"
            ),
            web_search_exam_prep_max_selected_results=int(
                os.getenv("WEB_SEARCH_EXAM_PREP_MAX_SELECTED_RESULTS", "2")
            ),
            web_search_require_official_for_exam_updates=(
                os.getenv("WEB_SEARCH_REQUIRE_OFFICIAL_FOR_EXAM_UPDATES", "true").lower()
                == "true"
            ),
            web_search_require_trusted_for_current_affairs=(
                os.getenv("WEB_SEARCH_REQUIRE_TRUSTED_FOR_CURRENT_AFFAIRS", "true").lower()
                == "true"
            ),
            web_search_min_trusted_results=int(
                os.getenv("WEB_SEARCH_MIN_TRUSTED_RESULTS", "1")
            ),
            web_search_default_recent_days=int(
                os.getenv("WEB_SEARCH_DEFAULT_RECENT_DAYS", "30")
            ),
            web_search_search_depth=os.getenv("WEB_SEARCH_SEARCH_DEPTH", "basic"),
            web_search_enable_extract=(
                os.getenv("WEB_SEARCH_ENABLE_EXTRACT", "false").lower() == "true"
            ),
            web_search_extract_max_urls=int(os.getenv("WEB_SEARCH_EXTRACT_MAX_URLS", "2")),
            web_search_extract_depth=os.getenv("WEB_SEARCH_EXTRACT_DEPTH", "basic"),
            web_search_allowed_domains=_parse_domain_list(
                os.getenv("WEB_SEARCH_ALLOWED_DOMAINS", "")
            ),
            web_search_blocked_domains=_parse_domain_list(
                os.getenv("WEB_SEARCH_BLOCKED_DOMAINS", "")
            ),
            web_search_trusted_domains=_parse_domain_list(
                os.getenv("WEB_SEARCH_TRUSTED_DOMAINS", "")
            ),
            answer_completion_marker=os.getenv("ANSWER_COMPLETION_MARKER", "<ANSWER_DONE>"),
            answer_continuation_enabled=(
                os.getenv("ANSWER_CONTINUATION_ENABLED", "true").lower() == "true"
            ),
            answer_continuation_max_attempts=int(
                os.getenv("ANSWER_CONTINUATION_MAX_ATTEMPTS", "1")
            ),
            answer_quality_validation_enabled=(
                os.getenv("ANSWER_QUALITY_VALIDATION_ENABLED", "true").lower() == "true"
            ),
            answer_quality_rewrite_enabled=(
                os.getenv("ANSWER_QUALITY_REWRITE_ENABLED", "true").lower() == "true"
            ),
            answer_quality_math_intermediate_max_chars=int(
                os.getenv("ANSWER_QUALITY_MATH_INTERMEDIATE_MAX_CHARS", "2200")
            ),
            answer_quality_max_rewrite_attempts=int(
                os.getenv("ANSWER_QUALITY_MAX_REWRITE_ATTEMPTS", "1")
            ),
            answer_quality_max_visible_steps=int(
                os.getenv("ANSWER_QUALITY_MAX_VISIBLE_STEPS", "8")
            ),
            answer_quality_max_display_math_blocks=int(
                os.getenv("ANSWER_QUALITY_MAX_DISPLAY_MATH_BLOCKS", "6")
            ),
            answer_quality_max_math_line_chars=int(
                os.getenv("ANSWER_QUALITY_MAX_MATH_LINE_CHARS", "300")
            ),
            gemini_api_key=os.getenv("GEMINI_API_KEY", ""),
            gemini_base_url=os.getenv("GEMINI_BASE_URL", ""),
            gemini_timeout_seconds=int(os.getenv("GEMINI_TIMEOUT_SECONDS", "30")),
            gemini_default_model=os.getenv("GEMINI_DEFAULT_MODEL", "gemini-2.5-flash-lite"),
            gemini_image_model=os.getenv("GEMINI_IMAGE_MODEL", "gemini-2.5-flash-lite"),
            gemini_text_model=os.getenv("GEMINI_TEXT_MODEL", "gemini-2.5-flash"),
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", ""),
            deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", ""),
            deepseek_timeout_seconds=int(os.getenv("DEEPSEEK_TIMEOUT_SECONDS", "60")),
            deepseek_default_model=os.getenv("DEEPSEEK_DEFAULT_MODEL", "deepseek-chat"),
            deepseek_reasoner_model=os.getenv("DEEPSEEK_REASONER_MODEL", "deepseek-reasoner"),
            llm_enable_gemini_routes=(
                os.getenv("LLM_ENABLE_GEMINI_ROUTES", "false").lower() == "true"
            ),
            llm_enable_deepseek_routes=(
                os.getenv("LLM_ENABLE_DEEPSEEK_ROUTES", "false").lower() == "true"
            ),
            azure_openai_deployment_gpt_4_1=os.getenv(
                "AZURE_OPENAI_DEPLOYMENT_GPT_4_1", "gpt-4.1"
            ),
            azure_openai_deployment_gpt_4_1_mini=os.getenv(
                "AZURE_OPENAI_DEPLOYMENT_GPT_4_1_MINI", "gpt-4.1-mini"
            ),
            azure_openai_deployment_gpt_5_4=os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT_5_4", ""),
            azure_openai_deployment_gpt_5_4_mini=os.getenv(
                "AZURE_OPENAI_DEPLOYMENT_GPT_5_4_MINI", ""
            ),
            azure_openai_deployment_gpt_5_5=os.getenv("AZURE_OPENAI_DEPLOYMENT_GPT_5_5", ""),
            azure_openai_deployment_o4_mini=os.getenv(
                "AZURE_OPENAI_DEPLOYMENT_O4_MINI", "o4-mini"
            ),
            azure_openai_deployment_o3=os.getenv("AZURE_OPENAI_DEPLOYMENT_O3", "o3"),
            azure_openai_send_reasoning_effort=(
                os.getenv("AZURE_OPENAI_SEND_REASONING_EFFORT", "false").lower() == "true"
            ),
            deepseek_advanced_model=os.getenv(
                "DEEPSEEK_ADVANCED_MODEL", "deepseek-reasoner"
            ),
            deepseek_v4pro_model=os.getenv("DEEPSEEK_V4PRO_MODEL", "deepseek-reasoner"),
            azure_deepseek_api_key=os.getenv("AZURE_DEEPSEEK_API_KEY", ""),
            azure_deepseek_endpoint=os.getenv("AZURE_DEEPSEEK_ENDPOINT", ""),
            azure_deepseek_api_version=os.getenv("AZURE_DEEPSEEK_API_VERSION", ""),
            azure_deepseek_timeout_seconds=int(
                os.getenv("AZURE_DEEPSEEK_TIMEOUT_SECONDS", "90")
            ),
            azure_deepseek_reasoner_deployment=os.getenv(
                "AZURE_DEEPSEEK_REASONER_DEPLOYMENT", ""
            ),
            azure_deepseek_chat_deployment=os.getenv(
                "AZURE_DEEPSEEK_CHAT_DEPLOYMENT", ""
            ),
            azure_deepseek_advanced_deployment=os.getenv(
                "AZURE_DEEPSEEK_ADVANCED_DEPLOYMENT", ""
            ),
        )
    return _settings


def get_llm_role_config(role: str, settings: Settings | None = None):  # -> LlmRoleConfig
    """Return the LlmRoleConfig for a named role (legacy model_router path only).

    Orchestrated doubt solver (ENABLE_ORCHESTRATED_DOUBT_SOLVER=true) uses YAML
    llm_routes.yaml + model_registry.yaml — not this function.

    LLM_ROLE_CONFIG_JSON values may be:
      - model alias strings (preferred) resolved via model_registry.yaml
      - legacy inline provider config objects (deprecated)

    When ENABLE_REAL_LLM=false (default), any role not found in the config map
    falls back to a local mock config so development works without credentials.

    Args:
        role:     The named role to look up (e.g. 'doubt_solver_classifier').
        settings: Optional Settings override for testing.

    Returns:
        LlmRoleConfig for the role.

    Raises:
        LlmConfigurationError: If the JSON is malformed, or if ENABLE_REAL_LLM=true
                               and no config exists for the given role.
    """
    from schemas.llm import LlmRoleConfig  # noqa: PLC0415
    from services.llm.llm_role_config import (  # noqa: PLC0415
        parse_role_config_map,
        resolve_llm_role_config,
        validate_role_config_aliases,
    )
    from services.llm.providers.errors import LlmConfigurationError  # noqa: PLC0415

    s = settings or get_settings()
    role_map = parse_role_config_map(s.llm_role_config_json)

    if role in role_map:
        if s.enable_real_llm:
            validate_role_config_aliases(role_map)
        config, _source = resolve_llm_role_config(role, role_map)
        return config

    if s.enable_real_llm:
        raise LlmConfigurationError(
            f"ENABLE_REAL_LLM=true but no role config found for role={role!r}. "
            "Add an entry to LLM_ROLE_CONFIG_JSON for this role."
        )

    return LlmRoleConfig(
        provider="mock",
        model_label="local-mock",
        supports_streaming=True,
    )
