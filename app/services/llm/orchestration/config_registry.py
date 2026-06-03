"""
app/services/llm_orchestration/config_registry.py
---------------------------------------------------
LLM orchestration config registry.

Responsibilities:
- Load the split config files (llm_routes.yaml, model_registry.yaml,
  provider_profiles.yaml) once per registry instance.
- Validate each file with strict Pydantic v2 schemas.
- Resolve route inheritance at build time (zero per-request overhead).
- Compile route_map, model_map, and provider_profile_map as in-memory dicts.
- Cross-validate all model aliases, provider profile references, and fallback
  symbols at build time.
- Expose read-only lookup methods for the route resolver.

Backward compatibility:
- The ``yaml_path`` constructor parameter still loads the old combined-format
  YAML used by tests that inject inline YAML fixtures.

Performance:
- YAML is parsed once per process (module-level singleton).
- All lookups at request time are plain dict[tuple → value] operations.
- No I/O at request time.

Security:
- Only yaml.safe_load() is used — never yaml.load().
- Logs config_version, route count, and model count only (no YAML contents).
- No secret values flow through this module.
"""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from schemas.llm_routing import (
    LlmOrchestrationConfig,
    LlmRoutesConfig,
    ModelConfig,
    ModelRegistryConfig,
    ProviderProfile,
    ProviderProfilesConfig,
    ResolvedRouteEntry,
    RouteEntry,
)
from services.llm.orchestration.errors import (
    LlmConfigLoadError,
    LlmConfigValidationError,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# app/ directory — two levels above this file (services/llm_orchestration/)
APP_DIR: Path = Path(__file__).resolve().parents[3]

# Production split-file paths (Part 7.1)
DEFAULT_ROUTES_PATH: Path = APP_DIR / "config" / "llm" / "llm_routes.yaml"
DEFAULT_MODEL_REGISTRY_PATH: Path = APP_DIR / "config" / "llm" / "model_registry.yaml"
DEFAULT_PROVIDER_PROFILES_PATH: Path = APP_DIR / "config" / "llm" / "provider_profiles.yaml"

# DEPRECATED: old combined-file path — kept for backward compat with test fixtures
DEFAULT_LLM_CONFIG_PATH: Path = APP_DIR / "config" / "llm" / "llm_orchestration.yaml"

# ---------------------------------------------------------------------------
# Allowed fallback symbols
# ---------------------------------------------------------------------------

_ALLOWED_FALLBACK_SYMBOLS: frozenset[str] = frozenset(
    {"basic", "intermediate", "advanced", "default", "general_default", "safe_mock"}
)

_DIFFICULTY_SYMBOLS: frozenset[str] = frozenset({"basic", "intermediate", "advanced", "default"})


# ---------------------------------------------------------------------------
# Registry class
# ---------------------------------------------------------------------------


class LlmConfigRegistry:
    """Compiled in-memory registry for the LLM orchestration config.

    Args:
        yaml_path: DEPRECATED.  Path to the old combined-format YAML config
                   (routes + models + provider_profiles in one file).  Kept
                   for backward compatibility with tests that inject inline
                   YAML fixtures.  Ignored when split paths are used.
        routes_path: Path to llm_routes.yaml.  Defaults to
                     DEFAULT_ROUTES_PATH.
        model_registry_path: Path to model_registry.yaml.  Defaults to
                             DEFAULT_MODEL_REGISTRY_PATH.
        provider_profiles_path: Path to provider_profiles.yaml.  Defaults
                                to DEFAULT_PROVIDER_PROFILES_PATH.

    When ``yaml_path`` is provided the registry loads from the combined
    format (backward compat).  Otherwise it loads from the three split files.
    """

    def __init__(
        self,
        yaml_path: Path | None = None,
        *,
        routes_path: Path | None = None,
        model_registry_path: Path | None = None,
        provider_profiles_path: Path | None = None,
    ) -> None:
        self._route_map: dict[tuple[str, str, str], ResolvedRouteEntry] = {}
        self._model_map: dict[str, ModelConfig] = {}
        self._provider_profile_map: dict[str, ProviderProfile] = {}
        self._config_version: int = 1

        if yaml_path is not None:
            # DEPRECATED path: load from a single combined YAML (used by tests).
            config = self._load_combined(yaml_path)
            self._config_version = config.version
            self._build(
                routes=config.routes,
                models=config.models,
                provider_profiles=config.provider_profiles,
            )
        else:
            # Primary path: load from three separate YAML files.
            effective_routes_path = routes_path or DEFAULT_ROUTES_PATH
            effective_model_registry_path = model_registry_path or DEFAULT_MODEL_REGISTRY_PATH
            effective_provider_profiles_path = (
                provider_profiles_path or DEFAULT_PROVIDER_PROFILES_PATH
            )

            routes_cfg = self._load_routes_file(effective_routes_path)
            model_registry_cfg = self._load_model_registry_file(effective_model_registry_path)
            provider_profiles_cfg = self._load_provider_profiles_file(
                effective_provider_profiles_path
            )

            self._config_version = routes_cfg.version
            self._build(
                routes=routes_cfg.routes,
                models=model_registry_cfg.models,
                provider_profiles=provider_profiles_cfg.provider_profiles,
            )

    # ------------------------------------------------------------------
    # Public read-only accessors
    # ------------------------------------------------------------------

    @property
    def config_version(self) -> int:
        return self._config_version

    @property
    def route_map(self) -> dict[tuple[str, str, str], ResolvedRouteEntry]:
        return self._route_map

    @property
    def model_map(self) -> dict[str, ModelConfig]:
        return self._model_map

    @property
    def provider_profile_map(self) -> dict[str, ProviderProfile]:
        return self._provider_profile_map

    def get_route(
        self, subject: str, task_role: str, difficulty: str
    ) -> ResolvedRouteEntry | None:
        """Return the compiled route entry or None if not found."""
        return self._route_map.get((subject, task_role, difficulty))

    def get_model(self, alias: str) -> ModelConfig | None:
        """Return the model config for a given alias or None."""
        return self._model_map.get(alias)

    def get_provider_profile(self, name: str) -> ProviderProfile | None:
        """Return the provider profile for a given name or None."""
        return self._provider_profile_map.get(name)

    # ------------------------------------------------------------------
    # Load and validate
    # ------------------------------------------------------------------

    def _load_yaml_file(self, path: Path) -> Any:
        """Read and parse a YAML file.  Security: uses yaml.safe_load only."""
        if not path.exists():
            raise LlmConfigLoadError(
                f"LLM config file not found: {path}"
            )
        try:
            with path.open("r", encoding="utf-8") as fh:
                # SECURITY: yaml.safe_load only — never yaml.load()
                raw: Any = yaml.safe_load(fh)
        except OSError as exc:
            raise LlmConfigLoadError(
                f"Cannot read LLM config file: {path}"
            ) from exc
        except yaml.YAMLError as exc:
            raise LlmConfigLoadError(
                f"YAML parse error in LLM config file: {exc}"
            ) from exc

        if not isinstance(raw, dict):
            raise LlmConfigLoadError(
                f"LLM config must be a YAML mapping at the top level: {path}"
            )
        return raw

    def _load_combined(self, yaml_path: Path) -> LlmOrchestrationConfig:
        """Load from the old combined YAML format.  DEPRECATED."""
        raw = self._load_yaml_file(yaml_path)
        try:
            config = LlmOrchestrationConfig.model_validate(raw)
        except ValidationError as exc:
            raise LlmConfigValidationError(
                f"LLM orchestration config failed Pydantic validation: {exc}"
            ) from exc
        logger.info(
            "llm_config_registry  loaded(combined)  version=%d  subjects=%d",
            config.version,
            len(config.routes),
        )
        return config

    def _load_routes_file(self, path: Path) -> LlmRoutesConfig:
        """Load and validate llm_routes.yaml."""
        raw = self._load_yaml_file(path)
        try:
            config = LlmRoutesConfig.model_validate(raw)
        except ValidationError as exc:
            raise LlmConfigValidationError(
                f"Routes config failed Pydantic validation ({path}): {exc}"
            ) from exc
        logger.info(
            "llm_config_registry  routes  loaded  version=%d  subjects=%d",
            config.version,
            len(config.routes),
        )
        return config

    def _load_model_registry_file(self, path: Path) -> ModelRegistryConfig:
        """Load and validate model_registry.yaml."""
        raw = self._load_yaml_file(path)
        try:
            config = ModelRegistryConfig.model_validate(raw)
        except ValidationError as exc:
            raise LlmConfigValidationError(
                f"Model registry config failed Pydantic validation ({path}): {exc}"
            ) from exc
        logger.info(
            "llm_config_registry  models  loaded  count=%d",
            len(config.models),
        )
        return config

    def _load_provider_profiles_file(self, path: Path) -> ProviderProfilesConfig:
        """Load and validate provider_profiles.yaml."""
        raw = self._load_yaml_file(path)
        try:
            config = ProviderProfilesConfig.model_validate(raw)
        except ValidationError as exc:
            raise LlmConfigValidationError(
                f"Provider profiles config failed Pydantic validation ({path}): {exc}"
            ) from exc
        logger.info(
            "llm_config_registry  profiles  loaded  count=%d",
            len(config.provider_profiles),
        )
        return config

    # ------------------------------------------------------------------
    # Build compiled maps
    # ------------------------------------------------------------------

    def _build(
        self,
        *,
        routes: dict[str, dict[str, dict[str, RouteEntry]]],
        models: dict[str, ModelConfig],
        provider_profiles: dict[str, ProviderProfile],
    ) -> None:
        """Resolve inheritance, compile maps, and cross-validate."""
        from services.llm.orchestration.model_registry_env import (  # noqa: PLC0415
            apply_model_registry_env_overrides,
        )

        self._model_map = dict(models)
        apply_model_registry_env_overrides(self._model_map)
        self._provider_profile_map = dict(provider_profiles)
        self._build_route_map(routes)
        self._cross_validate()

        logger.info(
            "llm_config_registry  compiled  routes=%d  models=%d  profiles=%d",
            len(self._route_map),
            len(self._model_map),
            len(self._provider_profile_map),
        )

    def _build_route_map(
        self,
        routes: dict[str, dict[str, dict[str, RouteEntry]]],
    ) -> None:
        """Resolve inheritance for every route entry and populate route_map."""
        for subject, roles in routes.items():
            for task_role, difficulties in roles.items():
                self._resolve_role_routes(subject, task_role, difficulties)

    def _resolve_role_routes(
        self,
        subject: str,
        task_role: str,
        difficulties: dict[str, RouteEntry],
    ) -> None:
        """Resolve inheritance for all difficulty levels within one (subject, task_role)."""
        default_entry = difficulties.get("default")

        for difficulty, entry in difficulties.items():
            resolved = self._resolve_entry(
                subject=subject,
                task_role=task_role,
                difficulty=difficulty,
                entry=entry,
                difficulties=difficulties,
                default_entry=default_entry,
            )
            self._validate_fallback_no_self(subject, task_role, difficulty, resolved.fallback)
            self._route_map[(subject, task_role, difficulty)] = resolved

    def _resolve_entry(
        self,
        subject: str,
        task_role: str,
        difficulty: str,
        entry: RouteEntry,
        difficulties: dict[str, RouteEntry],
        default_entry: RouteEntry | None,
    ) -> ResolvedRouteEntry:
        """Apply inheritance rules and return a ResolvedRouteEntry.

        Inheritance rules:
        - child scalar (model, prompt, temperature, max_tokens) overrides parent.
        - overlays = parent.overlays + child.overlays (concatenated).
        - provider_options = shallow merge, child keys override same parent keys.
        - fallback = child.fallback if child provided it (non-empty), else parent.fallback.
        """
        parent: RouteEntry | None = None

        if entry.inherits is not None:
            parent_key = entry.inherits
            parent = difficulties.get(parent_key)
            if parent is None:
                raise LlmConfigValidationError(
                    f"Route {subject}.{task_role}.{difficulty} inherits from "
                    f"'{parent_key}' which does not exist in the same task_role block."
                )
        elif difficulty != "default" and default_entry is not None:
            # Non-default entries without explicit `inherits` still fall back on
            # the default for missing required fields.
            parent = default_entry

        # Resolve each field using inheritance rules
        resolved_model = entry.model or (parent.model if parent else None)
        resolved_prompt = entry.prompt or (parent.prompt if parent else None)
        resolved_temperature = (
            entry.temperature if entry.temperature is not None
            else (parent.temperature if parent else None)
        )
        resolved_max_tokens = (
            entry.max_tokens if entry.max_tokens is not None
            else (parent.max_tokens if parent else None)
        )

        # overlays: concatenate parent + child
        parent_overlays = parent.overlays if parent else []
        resolved_overlays = parent_overlays + entry.overlays

        # intent_overlays: shallow merge keyed by intent; child keys override parent
        parent_intent_overlays = parent.intent_overlays if parent else {}
        resolved_intent_overlays: dict[str, list[str]] = {}
        for k in set(parent_intent_overlays) | set(entry.intent_overlays):
            child_paths = entry.intent_overlays.get(k)
            parent_paths = parent_intent_overlays.get(k, [])
            resolved_intent_overlays[k] = child_paths if child_paths is not None else parent_paths

        # provider_options: shallow merge, child overrides
        parent_opts = parent.provider_options if parent else {}
        resolved_opts = {**parent_opts, **entry.provider_options}

        # fallback: child wins if non-empty, else parent
        if entry.fallback:
            resolved_fallback = entry.fallback
        elif parent and parent.fallback:
            resolved_fallback = parent.fallback
        else:
            resolved_fallback = []

        # Validate required fields are now present
        if resolved_model is None:
            raise LlmConfigValidationError(
                f"Route {subject}.{task_role}.{difficulty} has no 'model' after "
                "inheritance resolution."
            )
        if resolved_prompt is None:
            raise LlmConfigValidationError(
                f"Route {subject}.{task_role}.{difficulty} has no 'prompt' after "
                "inheritance resolution."
            )
        if resolved_temperature is None:
            raise LlmConfigValidationError(
                f"Route {subject}.{task_role}.{difficulty} has no 'temperature' after "
                "inheritance resolution."
            )
        if resolved_max_tokens is None:
            raise LlmConfigValidationError(
                f"Route {subject}.{task_role}.{difficulty} has no 'max_tokens' after "
                "inheritance resolution."
            )

        try:
            return ResolvedRouteEntry(
                model=resolved_model,
                prompt=resolved_prompt,
                overlays=resolved_overlays,
                intent_overlays=resolved_intent_overlays,
                temperature=resolved_temperature,
                max_tokens=resolved_max_tokens,
                provider_options=resolved_opts,
                fallback=resolved_fallback,
            )
        except ValidationError as exc:
            raise LlmConfigValidationError(
                f"Route {subject}.{task_role}.{difficulty} failed validation after "
                f"inheritance resolution: {exc}"
            ) from exc

    def _validate_fallback_no_self(
        self, subject: str, task_role: str, difficulty: str, fallback: list[str]
    ) -> None:
        """Reject self-referencing fallback symbols."""
        for symbol in fallback:
            # A difficulty-level symbol that matches the current difficulty is a self-ref
            if symbol in _DIFFICULTY_SYMBOLS and symbol == difficulty:
                raise LlmConfigValidationError(
                    f"Route {subject}.{task_role}.{difficulty} has a self-referencing "
                    f"fallback symbol '{symbol}'."
                )

    # ------------------------------------------------------------------
    # Cross-validation
    # ------------------------------------------------------------------

    def _cross_validate(self) -> None:
        """Validate all referenced aliases and profiles exist."""
        self._validate_model_references()
        self._validate_provider_profile_references()
        self._validate_provider_consistency()
        self._validate_safe_mock_exists()
        self._validate_fallback_model_aliases()
        self._warn_placeholder_deployments()

    def _validate_model_references(self) -> None:
        """Every model alias referenced in routes must exist in model_map."""
        for (subject, task_role, difficulty), route in self._route_map.items():
            if route.model not in self._model_map:
                raise LlmConfigValidationError(
                    f"Route {subject}.{task_role}.{difficulty} references model "
                    f"'{route.model}' which does not exist in the models catalog."
                )

    def _validate_provider_profile_references(self) -> None:
        """Every provider_profile reference in models must exist in provider_profile_map."""
        for alias, model_cfg in self._model_map.items():
            if model_cfg.provider_profile not in self._provider_profile_map:
                raise LlmConfigValidationError(
                    f"Model '{alias}' references provider_profile "
                    f"'{model_cfg.provider_profile}' which does not exist in "
                    "provider_profiles."
                )

    def _validate_provider_consistency(self) -> None:
        """Each model's provider must match the provider of its provider_profile."""
        for alias, model_cfg in self._model_map.items():
            profile = self._provider_profile_map.get(model_cfg.provider_profile)
            if profile is None:
                # Already caught by _validate_provider_profile_references
                continue
            if model_cfg.provider != profile.provider:
                raise LlmConfigValidationError(
                    f"Model '{alias}' has provider='{model_cfg.provider}' but its "
                    f"provider_profile '{model_cfg.provider_profile}' has "
                    f"provider='{profile.provider}'. Provider must match."
                )

    def _validate_safe_mock_exists(self) -> None:
        """safe_mock model must always be present for the nuclear fallback path."""
        if "safe_mock" not in self._model_map:
            raise LlmConfigValidationError(
                "The 'safe_mock' model alias must be present in the models catalog."
            )

    def _validate_fallback_model_aliases(self) -> None:
        """Cross-validate all fallback_models entries in the model catalog.

        Checks:
        1. Every fallback alias exists in model_map.
        2. No self-reference (alias cannot be its own fallback).
        3. Provider profile of each fallback alias exists.
        4. No cyclic fallback chains (max depth 3 via BFS).
        """
        _MAX_FALLBACK_DEPTH = 3  # noqa: N806

        for alias, model_cfg in self._model_map.items():
            fallback_models = getattr(model_cfg, "fallback_models", None) or []
            if not fallback_models:
                continue

            for fallback_alias in fallback_models:
                # Check 1: fallback alias must exist
                if fallback_alias not in self._model_map:
                    raise LlmConfigValidationError(
                        f"Model '{alias}' has fallback_model '{fallback_alias}' "
                        "which does not exist in the models catalog."
                    )

                # Check 2: no self-reference
                if fallback_alias == alias:
                    raise LlmConfigValidationError(
                        f"Model '{alias}' cannot list itself in fallback_models."
                    )

                # Check 3: fallback's provider_profile must exist
                fallback_cfg = self._model_map[fallback_alias]
                if fallback_cfg.provider_profile not in self._provider_profile_map:
                    raise LlmConfigValidationError(
                        f"Fallback model '{fallback_alias}' (used by '{alias}') "
                        f"references provider_profile '{fallback_cfg.provider_profile}' "
                        "which does not exist in provider_profiles."
                    )

        # Check 4: no cycles (BFS from each alias up to max depth)
        for start_alias in self._model_map:
            self._check_no_fallback_cycle(start_alias, _MAX_FALLBACK_DEPTH)

    def _check_no_fallback_cycle(self, start_alias: str, max_depth: int) -> None:
        """BFS to detect cycles and enforce max fallback depth."""
        from collections import deque  # noqa: PLC0415

        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(start_alias, 0)])

        while queue:
            alias, depth = queue.popleft()
            if alias in visited:
                raise LlmConfigValidationError(
                    f"Cyclic fallback chain detected starting from '{start_alias}': "
                    f"'{alias}' appears more than once in the chain."
                )
            visited.add(alias)

            if depth >= max_depth:
                continue

            model_cfg = self._model_map.get(alias)
            if model_cfg is None:
                continue
            for fallback_alias in (getattr(model_cfg, "fallback_models", None) or []):
                if fallback_alias in self._model_map:
                    queue.append((fallback_alias, depth + 1))

    _PLACEHOLDER_PREFIXES: tuple[str, ...] = ("YOUR_", "TODO", "REPLACE_ME", "PLACEHOLDER")

    def _warn_placeholder_deployments(self) -> None:
        """Emit a startup warning for any Azure deployment name that looks like a placeholder.

        This is a warning (not an error) so the service remains startable in
        local/development environments where placeholders are expected until
        real deployment names are configured.

        Operators should replace all placeholder names before running in production.
        """
        import logging  # noqa: PLC0415

        _log = logging.getLogger(__name__)
        for alias, model_cfg in self._model_map.items():
            if model_cfg.provider == "azure_openai":
                dep = getattr(model_cfg, "deployment", None) or ""
                if any(dep.upper().startswith(p) for p in self._PLACEHOLDER_PREFIXES):
                    _log.warning(
                        "config_registry: model '%s' has a placeholder Azure deployment "
                        "name '%s'. Replace with a real deployment name before "
                        "running in production.",
                        alias,
                        dep,
                    )

    def _active_route_model_aliases(self) -> set[str]:
        """Model aliases referenced by production routes (excludes optional test difficulties)."""
        optional_test_difficulties = frozenset({
            "gemini_test",
            "deepseek_test",
            "deepseek_azure_test",
            "cat_advanced",
            "sbi_po_complex",
            "cat_lrdi",
        })
        aliases: set[str] = set()
        for (_subject, _task_role, difficulty), route in self._route_map.items():
            if difficulty in optional_test_difficulties:
                continue
            aliases.add(route.model)
        return aliases

    def validate_real_mode_deployments(self) -> None:
        """Raise if any *active-route* Azure model has empty/placeholder deployment.

        Only models referenced by llm_routes.yaml are checked. Optional catalog
        aliases (e.g. openai_gpt_5_4 when env unset) may remain inactive with
        blank deployments without blocking startup.

        Active routes using ``azure_deepseek`` profile must also have non-empty
        AZURE_DEEPSEEK_API_KEY, AZURE_DEEPSEEK_ENDPOINT, and (classic mode)
        AZURE_DEEPSEEK_API_VERSION env vars.

        Security: error messages include only model alias and deployment name.
        """
        import os  # noqa: PLC0415

        errors: list[str] = []
        active_aliases = self._active_route_model_aliases()

        for alias in sorted(active_aliases):
            model_cfg = self._model_map.get(alias)
            if model_cfg is None:
                continue
            if model_cfg.provider == "azure_openai":
                dep = getattr(model_cfg, "deployment", None) or ""
                profile_name = getattr(model_cfg, "provider_profile", None) or ""
                if not dep:
                    if profile_name == "azure_deepseek":
                        errors.append(
                            f"Active route model_alias='{alias}' has empty Azure "
                            "DeepSeek deployment. Set the matching "
                            "AZURE_DEEPSEEK_*_DEPLOYMENT env var or point the "
                            "route to a working alias."
                        )
                    else:
                        errors.append(
                            f"Active route model_alias='{alias}' has empty Azure "
                            "deployment. Set the matching AZURE_OPENAI_DEPLOYMENT_* "
                            "env var or point the route to a working alias."
                        )
                elif any(
                    dep.upper().startswith(p) or p in dep.upper()
                    for p in self._PLACEHOLDER_PREFIXES
                ):
                    errors.append(
                        f"Active route model_alias='{alias}' has placeholder "
                        "Azure deployment. Replace with a real deployment name."
                    )
                if profile_name == "azure_deepseek":
                    for env_var in (
                        "AZURE_DEEPSEEK_API_KEY",
                        "AZURE_DEEPSEEK_ENDPOINT",
                        "AZURE_DEEPSEEK_API_VERSION",
                    ):
                        if not os.getenv(env_var, "").strip():
                            errors.append(
                                f"Active route model_alias='{alias}' uses "
                                f"azure_deepseek profile but {env_var} is not set."
                            )
            elif model_cfg.provider == "openai":
                mid = getattr(model_cfg, "model_id", None) or ""
                if any(
                    mid.upper().startswith(p) or p in mid.upper()
                    for p in self._PLACEHOLDER_PREFIXES
                ):
                    errors.append(
                        f"Active route model_alias='{alias}' has placeholder "
                        "model_id. Replace in model_registry.yaml."
                    )

        if errors:
            raise LlmConfigValidationError(
                "Real-mode deployment preflight failed for active routes. "
                "Set ENABLE_REAL_LLM=false for local/mock operation, or configure "
                "deployments before going live. Affected: " + "; ".join(errors)
            )


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_registry: LlmConfigRegistry | None = None
_registry_lock = threading.Lock()


def get_registry(yaml_path: Path | None = None) -> LlmConfigRegistry:
    """Return the module-level singleton LlmConfigRegistry.

    Thread-safe lazy initialization.

    Pass ``yaml_path`` only to override with the old combined-format YAML
    (e.g. in tests that inject inline YAML fixtures).  Calling with a non-None
    ``yaml_path`` always creates a fresh registry (not cached).

    With no arguments the singleton is initialised from the three production
    split files (llm_routes.yaml, model_registry.yaml, provider_profiles.yaml).
    """
    global _registry  # noqa: PLW0603

    if yaml_path is not None:
        # Test override: always fresh, never cached.
        return LlmConfigRegistry(yaml_path=yaml_path)

    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = LlmConfigRegistry()
    return _registry


def reset_registry() -> None:
    """Reset the module-level singleton.  Use in tests only."""
    global _registry  # noqa: PLW0603
    with _registry_lock:
        _registry = None
