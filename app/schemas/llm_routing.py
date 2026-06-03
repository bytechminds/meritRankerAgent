"""
app/schemas/llm_routing.py
--------------------------
Pydantic v2 schemas for the LLM orchestration routing layer (Part 1).

These schemas validate the YAML config, model catalog, provider profiles, and
the runtime route resolver inputs/outputs.

Public types:
    ProviderName          — allowed provider names
    TaskRole              — task role enum
    DifficultyLevel       — difficulty level enum
    SubjectName           — known subject areas
    CostTier              — model cost classification
    CapabilityLevel       — model capability classification
    RouteEntry            — raw YAML route entry (pre-inheritance)
    ResolvedRouteEntry    — post-inheritance route (all fields required)
    ModelConfig           — model catalog entry
    ProviderProfile       — provider connection metadata (no secrets)
    LlmRoutesConfig       — schema for llm_routes.yaml
    ModelRegistryConfig   — schema for model_registry.yaml
    ProviderProfilesConfig — schema for provider_profiles.yaml
    LlmOrchestrationConfig — combined config schema (DEPRECATED)
    RouteRequest          — resolver input
    FallbackAttempt       — one step in a fallback chain
    RouteDecision         — resolver output (no credentials)

Security: ProviderProfile fields are validated to contain only env var names,
never actual secret values.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# Enums (plain str subclasses, not Enum, to keep them JSON-serializable and
# Pydantic-friendly without extra wrappers)
# ---------------------------------------------------------------------------

ProviderName = Literal["gemini", "azure_openai", "openai", "mock", "deepseek"]

# Azure API mode — determines how the adapter forms the HTTP request.
# Set in provider_profiles.yaml.  Defaults to azure_deployment_chat_completions
# for backward compatibility with existing profiles that omit the field.
#
# azure_deployment_chat_completions:
#   Classic Azure OpenAI deployment endpoint.
#   endpoint must be https://<resource>.openai.azure.com (NOT ending /openai/v1).
#   Adapter uses AzureOpenAI(azure_endpoint=...) + deployment path.
#   api_version required.
#
# azure_openai_v1:
#   OpenAI-compatible v1 API.
#   base_url (or endpoint) must end with /openai/v1/ after normalisation.
#   Adapter uses OpenAI(base_url=...) — no deployment path segment.
#   Passes deployment as model= parameter.
#   api_version not sent by our code (SDK handles it at base_url level).
AzureApiMode = Literal["azure_deployment_chat_completions", "azure_openai_v1"]

# Part 1: only `generator` is fully implemented.
# Other roles are present in the enum for future use — no routes or behavior
# exists for them in Part 1.
TaskRole = Literal[
    "classifier",
    "classifier_strong",
    "planner",
    "generator",
    "formatter",
    "verifier",
    "visual_formatter",
]

DifficultyLevel = Literal["default", "basic", "intermediate", "advanced"]

SubjectName = Literal["math", "reasoning", "english", "general"]

CostTier = Literal["none", "low", "medium", "high"]

CapabilityLevel = Literal["low", "medium", "high", "very_high"]

ReasoningEffort = Literal["none", "low", "medium", "high"]

TokenBudgetParam = Literal["max_tokens", "max_completion_tokens", "max_output_tokens"]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Allowed fallback symbols in Part 1
_ALLOWED_FALLBACK_SYMBOLS: frozenset[str] = frozenset(
    {"basic", "intermediate", "advanced", "default", "general_default", "safe_mock"}
)

# Allowed keys in intent_overlays map.
# Expanding beyond these four requires an explicit schema change.
_ALLOWED_INTENT_OVERLAY_KEYS: frozenset[str] = frozenset(
    {"solve", "explain", "practice", "visualize"}
)

# Keys that must never appear in a route entry — they belong in model_registry.yaml
_ROUTE_FORBIDDEN_KEYS: frozenset[str] = frozenset(
    {
        "provider",
        "provider_profile",
        "model_id",
        "deployment",
        "api_key_env",
        "endpoint_env",
        "credential_ref",
    }
)

# Detect obvious secret-like values in provider profile fields.
# This is not exhaustive — it catches common accidental mistakes.
_SECRET_PATTERN = re.compile(r"(^sk-|^AIza|^AKIA|-----BEGIN)", re.IGNORECASE)

# Environment variable names must follow SCREAMING_SNAKE_CASE.
_ENV_VAR_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")

# Max length for path strings to prevent unreasonable values.
_MAX_PATH_LEN = 256

# ---------------------------------------------------------------------------
# Prompt path safety validator (reused by RouteEntry)
# ---------------------------------------------------------------------------


def _validate_prompt_path(v: str | None, field_name: str = "path") -> str | None:
    """Validate that a prompt file path is safe for Part 1 (no file existence check)."""
    if v is None:
        return v
    if len(v) > _MAX_PATH_LEN:
        raise ValueError(f"{field_name} exceeds max length of {_MAX_PATH_LEN}")
    if v.startswith("/"):
        raise ValueError(f"{field_name} must be a relative path, not absolute")
    if re.search(r"\.\.", v):
        raise ValueError(f"{field_name} must not contain '..'")
    if re.match(r"https?://", v, re.IGNORECASE):
        raise ValueError(f"{field_name} must not be a URL")
    if not v.endswith(".md"):
        raise ValueError(f"{field_name} must end with '.md'")
    return v


# ---------------------------------------------------------------------------
# RouteEntry — raw YAML route (pre-inheritance resolution)
# ---------------------------------------------------------------------------


class RouteEntry(BaseModel):
    """One entry in the routes YAML.  Some fields may be absent and resolved via
    `inherits` at registry build time.

    Security: route entries must NOT contain provider-specific fields such as
    model_id, deployment, provider_profile, or api_key_env.  Those belong in
    model_registry.yaml.
    """

    model: str | None = Field(default=None, description="Model alias from the models catalog.")
    prompt: str | None = Field(default=None, description="Relative path to prompt template.")
    overlays: list[str] = Field(
        default_factory=list,
        description="Additional prompt overlay paths.",
    )
    intent_overlays: dict[str, list[str]] = Field(
        default_factory=dict,
        description=(
            "Intent-specific prompt overlay paths. "
            "Keys must be one of: solve, explain, practice, visualize. "
            "Overlays are appended after route overlays when the request intent matches."
        ),
    )
    inherits: str | None = Field(
        default=None, description="Difficulty key to inherit base config from."
    )
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    max_tokens: int | None = Field(default=None, gt=0, le=8000)
    provider_options: dict[str, Any] = Field(default_factory=dict)
    fallback: list[str] = Field(default_factory=list, description="Ordered fallback chain symbols.")

    model_config = {"str_strip_whitespace": True}

    @model_validator(mode="before")
    @classmethod
    def reject_provider_fields(cls, v: Any) -> Any:
        """Reject provider-specific keys in route entries.

        Route config must reference model aliases only.  Provider details
        (model_id, deployment, provider_profile, credential fields) belong
        in model_registry.yaml, not in routes.
        """
        if isinstance(v, dict):
            forbidden_found = _ROUTE_FORBIDDEN_KEYS.intersection(v.keys())
            if forbidden_found:
                raise ValueError(
                    f"Route entry contains forbidden provider-specific fields: "
                    f"{sorted(forbidden_found)}. Provider fields belong in "
                    "model_registry.yaml, not in route config."
                )
        return v

    @field_validator("prompt")
    @classmethod
    def validate_prompt_path(cls, v: str | None) -> str | None:
        return _validate_prompt_path(v, "prompt")

    @field_validator("overlays")
    @classmethod
    def validate_overlay_paths(cls, v: list[str]) -> list[str]:
        for path in v:
            _validate_prompt_path(path, "overlay")
        return v

    @field_validator("intent_overlays")
    @classmethod
    def validate_intent_overlays(cls, v: dict[str, list[str]]) -> dict[str, list[str]]:
        unknown_keys = set(v.keys()) - _ALLOWED_INTENT_OVERLAY_KEYS
        if unknown_keys:
            raise ValueError(
                f"intent_overlays contains unknown intent keys: {sorted(unknown_keys)}. "
                f"Allowed keys: {sorted(_ALLOWED_INTENT_OVERLAY_KEYS)}"
            )
        for intent_key, paths in v.items():
            for path in paths:
                _validate_prompt_path(path, f"intent_overlays[{intent_key!r}]")
        return v

    @field_validator("fallback")
    @classmethod
    def validate_fallback_symbols(cls, v: list[str]) -> list[str]:
        for symbol in v:
            if symbol not in _ALLOWED_FALLBACK_SYMBOLS:
                raise ValueError(
                    f"Invalid fallback symbol {symbol!r}. "
                    f"Allowed: {sorted(_ALLOWED_FALLBACK_SYMBOLS)}"
                )
        return v


# ---------------------------------------------------------------------------
# ResolvedRouteEntry — post-inheritance (all required fields present)
# ---------------------------------------------------------------------------


class ResolvedRouteEntry(BaseModel):
    """A fully resolved route entry after inheritance and default merge.

    All fields are required — no `None` values allowed.  This is what gets
    stored in `route_map` at build time.
    """

    model: str = Field(description="Model alias.")
    prompt: str = Field(description="Relative path to prompt template.")
    overlays: list[str] = Field(default_factory=list)
    intent_overlays: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Intent-specific overlay paths keyed by normalized intent name.",
    )
    temperature: float = Field(ge=0.0, le=2.0)
    max_tokens: int = Field(gt=0, le=8000)
    provider_options: dict[str, Any] = Field(default_factory=dict)
    fallback: list[str] = Field(default_factory=list)

    @field_validator("prompt")
    @classmethod
    def validate_prompt_path(cls, v: str) -> str:
        result = _validate_prompt_path(v, "prompt")
        assert result is not None  # guaranteed non-None because field is required
        return result

    @field_validator("overlays")
    @classmethod
    def validate_overlay_paths(cls, v: list[str]) -> list[str]:
        for path in v:
            _validate_prompt_path(path, "overlay")
        return v

    @field_validator("intent_overlays")
    @classmethod
    def validate_intent_overlays(cls, v: dict[str, list[str]]) -> dict[str, list[str]]:
        unknown_keys = set(v.keys()) - _ALLOWED_INTENT_OVERLAY_KEYS
        if unknown_keys:
            raise ValueError(
                f"intent_overlays contains unknown intent keys: {sorted(unknown_keys)}. "
                f"Allowed keys: {sorted(_ALLOWED_INTENT_OVERLAY_KEYS)}"
            )
        for intent_key, paths in v.items():
            for path in paths:
                _validate_prompt_path(path, f"intent_overlays[{intent_key!r}]")
        return v


# ---------------------------------------------------------------------------
# ModelConfig — entry in the `models` catalog
# ---------------------------------------------------------------------------


class ModelConfig(BaseModel):
    """Model catalog entry.  Contains no secrets — only model metadata."""

    provider: ProviderName
    provider_profile: str = Field(min_length=1)
    model_id: str | None = Field(
        default=None, description="Provider model ID (required unless provider=mock)."
    )
    deployment: str | None = Field(
        default=None, description="Azure deployment name (alternative to model_id)."
    )
    description: str | None = Field(
        default=None, description="Human-readable description for docs."
    )
    model_label: str | None = Field(default=None, description="Human-readable label for logging.")
    cost_tier: CostTier | None = Field(default=None)
    supports_streaming: bool = False
    supports_thinking: bool = False
    supports_reasoning: bool = Field(
        default=False,
        description=(
            "Model supports provider reasoning/thinking mode. When true, routes may "
            "pass provider_options.thinking=true. Params are sent only when the "
            "adapter implements reasoning for this provider."
        ),
    )
    reasoning_effort: ReasoningEffort = Field(
        default="none",
        description=(
            "Suggested reasoning effort for models that support reasoning. Stored as "
            "metadata; adapters send this only when provider support is verified."
        ),
    )
    token_budget_param: TokenBudgetParam = Field(
        default="max_tokens",
        description=(
            "Provider API field for the route output token budget. Azure/OpenAI "
            "reasoning models require max_completion_tokens."
        ),
    )
    supports_temperature: bool = Field(
        default=True,
        description="When false, temperature is omitted from provider payloads.",
    )
    supports_top_p: bool = Field(
        default=False,
        description="When false, top_p is omitted from provider payloads.",
    )
    supports_penalties: bool = Field(
        default=False,
        description=(
            "When false, presence_penalty and frequency_penalty are omitted."
        ),
    )
    supports_logprobs: bool = Field(
        default=False,
        description="When false, logprobs, top_logprobs, and logit_bias are omitted.",
    )
    supports_stop: bool = Field(
        default=True,
        description="When false, stop sequences are omitted from provider payloads.",
    )
    send_reasoning_effort: bool = Field(
        default=False,
        description=(
            "When true and AZURE_OPENAI_SEND_REASONING_EFFORT is enabled, adapters "
            "may send reasoning_effort to providers that support it."
        ),
    )
    timeout_seconds: int = Field(ge=1, le=120)
    capabilities: dict[str, CapabilityLevel] = Field(
        default_factory=dict,
        description="Subject capability ratings. [NOT VERIFIED] — placeholder until live eval.",
    )
    fallback_models: list[str] = Field(
        default_factory=list,
        description=(
            "Ordered list of model aliases to try if this model fails with a "
            "controlled provider execution error.  Entries must be internal model "
            "aliases (not provider model_id or deployment names).  Max 3 entries."
        ),
    )

    @model_validator(mode="after")
    def require_model_id_or_deployment(self) -> ModelConfig:
        """model_id or deployment must be present unless provider is mock."""
        if self.provider != "mock" and self.model_id is None and self.deployment is None:
            raise ValueError(
                f"ModelConfig with provider={self.provider!r} must have "
                "either 'model_id' or 'deployment'."
            )
        return self

    @model_validator(mode="after")
    def align_reasoning_capabilities(self) -> ModelConfig:
        """When supports_reasoning is true, allow thinking provider option."""
        if self.supports_reasoning and not self.supports_thinking:
            return self.model_copy(update={"supports_thinking": True})
        return self

    @field_validator("fallback_models")
    @classmethod
    def validate_fallback_models(cls, v: list[str]) -> list[str]:
        """Validate fallback_models list contents (not cross-registry checks)."""
        if len(v) > 3:
            raise ValueError(
                f"fallback_models may contain at most 3 aliases (got {len(v)})."
            )
        for alias in v:
            if not alias or not alias.strip():
                raise ValueError("fallback_models entries must not be empty strings.")
            # Entries must look like internal snake_case aliases, not provider model IDs.
            # Provider model IDs typically contain slashes, dots, or start with
            # known prefixes (gpt-, gemini-, claude-).  We reject those patterns.
            if "/" in alias or alias.startswith(("gpt-", "gemini-", "claude-", "text-")):
                raise ValueError(
                    f"fallback_models entry {alias!r} looks like a provider model_id, "
                    "not an internal alias.  Use the internal alias name from "
                    "model_registry.yaml (e.g. 'math_basic_generator_openai_native')."
                )
            if "." in alias:
                raise ValueError(
                    f"fallback_models entry {alias!r} must not contain dots. "
                    "Use internal snake_case aliases only."
                )
        return v


# ---------------------------------------------------------------------------
# ProviderProfile — connection metadata (env var names only, never secrets)
# ---------------------------------------------------------------------------


def _check_not_secret(v: str | None, field_name: str) -> str | None:
    """Reject values that look like actual secret material."""
    if v is None:
        return v
    if _SECRET_PATTERN.search(v):
        raise ValueError(
            f"ProviderProfile.{field_name} looks like a real secret value "
            f"(matched pattern: sk-, AIza, AKIA, or '-----BEGIN'). "
            "Store only the environment variable NAME here, not the actual key."
        )
    return v


def _check_env_var_name(v: str | None, field_name: str) -> str | None:
    """Fields ending in _env must be valid SCREAMING_SNAKE_CASE env var names."""
    if v is None:
        return v
    if not _ENV_VAR_PATTERN.match(v):
        raise ValueError(
            f"ProviderProfile.{field_name}={v!r} is not a valid environment variable name. "
            "Must match ^[A-Z][A-Z0-9_]*$ (SCREAMING_SNAKE_CASE)."
        )
    return v


class ProviderProfile(BaseModel):
    """Provider connection metadata.

    Security contract: all string fields must be environment variable NAMES
    (SCREAMING_SNAKE_CASE) or safe reference strings — never actual secret values.

    For azure_openai provider, the ``azure_api_mode`` field selects the HTTP
    request style:
    - ``azure_deployment_chat_completions`` (default): classic deployment endpoint.
      Requires endpoint_env pointing to https://<resource>.openai.azure.com
      (must NOT end with /openai/v1). Requires api_version_env.
    - ``azure_openai_v1``: OpenAI-compatible base URL.
      Requires base_url_env (or endpoint_env) pointing to a URL ending in
      /openai/v1 or /openai/v1/. Does not append deployment path segment.
    """

    provider: ProviderName
    azure_api_mode: AzureApiMode = Field(
        default="azure_deployment_chat_completions",
        description=(
            "Azure HTTP request style.  Only used when provider=azure_openai. "
            "azure_deployment_chat_completions: classic deployment endpoint. "
            "azure_openai_v1: OpenAI-compatible /openai/v1/ base URL."
        ),
    )
    api_key_env: str | None = Field(
        default=None, description="Env var name holding the API key."
    )
    endpoint_env: str | None = Field(
        default=None, description="Env var name holding the endpoint URL."
    )
    api_version_env: str | None = Field(
        default=None, description="Env var name holding the API version string."
    )
    base_url_env: str | None = Field(
        default=None, description="Env var name holding a custom base URL (azure_openai_v1 mode)."
    )
    credential_ref: str | None = Field(
        default=None,
        max_length=256,
        description="AgentCore credential resource name or Secrets Manager ARN reference.",
    )
    optional_api_key: bool = Field(
        default=False,
        description=(
            "When true, a missing or blank api_key_env resolves to api_key=None "
            "instead of raising SecretNotFoundError. Used for optional providers."
        ),
    )
    optional_base_url: bool = Field(
        default=False,
        description=(
            "When true, a missing or blank base_url_env resolves to base_url=None "
            "instead of raising SecretNotFoundError."
        ),
    )
    optional_endpoint: bool = Field(
        default=False,
        description=(
            "When true, a missing or blank endpoint_env resolves to endpoint=None "
            "instead of raising SecretNotFoundError. Used for optional providers."
        ),
    )
    optional_api_version: bool = Field(
        default=False,
        description=(
            "When true, a missing or blank api_version_env resolves to api_version=None "
            "instead of raising SecretNotFoundError. Used for optional providers."
        ),
    )

    model_config = {"str_strip_whitespace": True}

    @field_validator("api_key_env")
    @classmethod
    def validate_api_key_env(cls, v: str | None) -> str | None:
        v = _check_not_secret(v, "api_key_env")
        return _check_env_var_name(v, "api_key_env")

    @field_validator("endpoint_env")
    @classmethod
    def validate_endpoint_env(cls, v: str | None) -> str | None:
        v = _check_not_secret(v, "endpoint_env")
        return _check_env_var_name(v, "endpoint_env")

    @field_validator("api_version_env")
    @classmethod
    def validate_api_version_env(cls, v: str | None) -> str | None:
        v = _check_not_secret(v, "api_version_env")
        return _check_env_var_name(v, "api_version_env")

    @field_validator("base_url_env")
    @classmethod
    def validate_base_url_env(cls, v: str | None) -> str | None:
        v = _check_not_secret(v, "base_url_env")
        return _check_env_var_name(v, "base_url_env")

    @field_validator("credential_ref")
    @classmethod
    def validate_credential_ref(cls, v: str | None) -> str | None:
        return _check_not_secret(v, "credential_ref")


# ---------------------------------------------------------------------------
# Split config schemas (Part 7.1 — three-file layout)
# ---------------------------------------------------------------------------


class LlmRoutesConfig(BaseModel):
    """Schema for llm_routes.yaml.  Contains only route definitions."""

    version: int = Field(ge=1, description="Config schema version.")
    routes: dict[str, dict[str, dict[str, RouteEntry]]] = Field(
        description="Routing table: subject → task_role → difficulty → RouteEntry."
    )


class ModelRegistryConfig(BaseModel):
    """Schema for model_registry.yaml.  Contains the flat model catalog."""

    version: int = Field(ge=1, description="Config schema version.")
    models: dict[str, ModelConfig] = Field(description="Model catalog keyed by alias.")


class ProviderProfilesConfig(BaseModel):
    """Schema for provider_profiles.yaml.  Contains provider connection metadata."""

    version: int = Field(ge=1, description="Config schema version.")
    provider_profiles: dict[str, ProviderProfile] = Field(
        description="Provider connection metadata keyed by profile name."
    )


# ---------------------------------------------------------------------------
# LlmOrchestrationConfig — combined top-level schema (DEPRECATED)
# ---------------------------------------------------------------------------


class LlmOrchestrationConfig(BaseModel):
    """Combined YAML config schema — DEPRECATED in favour of split files.

    routes structure: routes[subject][task_role][difficulty] = RouteEntry

    .. deprecated::
        Use LlmRoutesConfig + ModelRegistryConfig + ProviderProfilesConfig
        (loaded from llm_routes.yaml, model_registry.yaml, provider_profiles.yaml).
        This class is kept for backward compatibility with tests that inject
        inline combined-format YAML via ``LlmConfigRegistry(yaml_path=...)``.
    """

    version: int = Field(ge=1, description="Config schema version.")
    routes: dict[str, dict[str, dict[str, RouteEntry]]] = Field(
        description="Routing table: subject → task_role → difficulty → RouteEntry."
    )
    models: dict[str, ModelConfig] = Field(description="Model catalog keyed by alias.")
    provider_profiles: dict[str, ProviderProfile] = Field(
        description="Provider connection metadata keyed by profile name."
    )


# ---------------------------------------------------------------------------
# RouteRequest — runtime input to the route resolver
# ---------------------------------------------------------------------------


class RouteRequest(BaseModel):
    """Input to the route resolver at request time.

    No plan tier in Part 1.
    """

    request_id: str = Field(min_length=1, max_length=128)
    subject: str = Field(
        min_length=1, max_length=64, description="Raw subject string (will be normalized)."
    )
    task_role: TaskRole = Field(description="The task role to resolve a route for.")
    difficulty: str = Field(
        default="default",
        max_length=64,
        description="Raw difficulty string (will be normalized).",
    )
    intent: str | None = Field(default=None, max_length=128)
    exam: str | None = Field(default=None, max_length=128)

    model_config = {"str_strip_whitespace": True}


# ---------------------------------------------------------------------------
# FallbackAttempt — one step in a resolved fallback chain
# ---------------------------------------------------------------------------


class FallbackAttempt(BaseModel):
    """Represents a single fallback target resolved from a YAML fallback symbol."""

    kind: Literal["route", "model"]
    subject: str | None = None
    task_role: str | None = None
    difficulty: str | None = None
    model: str | None = None
    reason: str = Field(min_length=1, max_length=256)


# ---------------------------------------------------------------------------
# RouteDecision — resolver output (no credentials)
# ---------------------------------------------------------------------------


class RouteDecision(BaseModel):
    """Output of the route resolver.

    Security contract: this object contains NO credentials, no API keys, no
    model endpoints, no provider profile secret values.  The `model` field is
    a model alias (e.g. 'gemini_flash_light'), not the actual API model_id.
    """

    route_id: str = Field(
        min_length=1,
        description="Stable route identifier in format '{subject}.{task_role}.{difficulty}'.",
    )
    subject: str
    task_role: TaskRole
    difficulty: str
    intent: str | None = None
    exam: str | None = None
    model: str = Field(description="Model alias (not the actual provider model_id).")
    prompt: str = Field(description="Relative prompt file path.")
    overlays: list[str] = Field(default_factory=list)
    intent_overlays: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Intent-specific overlay paths from the route config.",
    )
    temperature: float
    max_tokens: int
    provider_options: dict[str, Any] = Field(default_factory=dict)
    fallback_attempts: list[FallbackAttempt] = Field(
        default_factory=list,
        description="Fallback targets resolved from the route's fallback list.",
    )
    route_source: Literal["exact", "subject_default", "general_default", "safe_mock"] = Field(
        description="Which fallback step produced this route decision."
    )
