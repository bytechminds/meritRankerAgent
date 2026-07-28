"""Unit and contract tests for the image-question classification layer."""

from __future__ import annotations

import base64
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest
from PIL import Image
from pydantic import ValidationError

import config as config_module
import services.image_question_classification.classifier as classifier_module
from graphs.image_question_classifier_node import ImageClassifierNodeInput, image_classifier_node
from observability.context import bind_request_context
from observability.llm_usage import (
    begin_llm_usage_collection,
    reset_llm_usage_collection,
    snapshot_llm_usage_records,
)
from schemas.doubt_solver import DoubtSolverRequest, QueryClassification
from schemas.image_input import ImageInput
from schemas.image_question_classification import (
    ImageClassificationResult,
    ImageClassificationStatus,
    ImageParseMetadata,
    ImageProviderOutput,
    VisualContext,
)
from schemas.llm_usage import ProviderTokenUsage
from services.image_question_classification.cache import ImageClassificationCache
from services.image_question_classification.classifier import ImageQuestionClassifier
from services.image_question_classification.errors import (
    ImageProviderResponseError,
    ImageProviderTemporaryError,
)
from services.image_question_classification.gemini_provider import (
    GeminiImageQuestionClassificationProvider,
)
from services.image_question_classification.image_loader import InlineImageLoader
from services.image_question_classification.image_validator import (
    ImageInputValidator,
    ImageValidationConfig,
)
from services.image_question_classification.prompt_builder import (
    build_image_classifier_prompt,
)


def _image_bytes(
    *,
    image_format: str = "PNG",
    size: tuple[int, int] = (320, 180),
) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, color="white").save(output, format=image_format)
    return output.getvalue()


def _animated_webp_bytes() -> bytes:
    output = io.BytesIO()
    first = Image.new("RGB", (320, 180), color="white")
    second = Image.new("RGB", (320, 180), color="black")
    first.save(
        output,
        format="WEBP",
        save_all=True,
        append_images=[second],
        duration=100,
        loop=0,
    )
    return output.getvalue()


def _image_input(
    content: bytes | None = None,
    *,
    mime_type: str = "image/png",
) -> ImageInput:
    return ImageInput(
        source="upload",
        mimeType=mime_type,
        base64=base64.b64encode(content or _image_bytes()).decode("ascii"),
    )


def _classification(
    *,
    subject: str = "math",
    confidence: float = 0.96,
) -> QueryClassification:
    return QueryClassification(
        intent="solve_question",
        subject=subject,
        topic="percentage",
        confidence=confidence,
        difficulty="intermediate",
        retrieval_need="concept_context",
    )


def _metadata(
    *,
    confidence: float = 0.95,
    visual_type: str = "none",
) -> ImageParseMetadata:
    return ImageParseMetadata(
        has_question=True,
        has_options=True,
        has_visual=visual_type != "none",
        visual_context=VisualContext(type=visual_type, confidence=confidence),
        extraction_confidence=confidence,
        classification_confidence=confidence,
        ignored_content_detected=True,
    )


def _provider_output(
    *,
    subject: str = "math",
    confidence: float = 0.95,
    visual_type: str = "none",
    provider_usage: ProviderTokenUsage | None = None,
) -> ImageProviderOutput:
    return ImageProviderOutput(
        status=ImageClassificationStatus.CLASSIFIED,
        normalized_query="What is 20% of 500? A. 50 B. 100 C. 150 D. 200",
        classification=_classification(subject=subject, confidence=confidence),
        image_parse_metadata=_metadata(
            confidence=confidence,
            visual_type=visual_type,
        ),
        provider_usage=provider_usage,
    )


def _gemini_wire_payload() -> dict:
    return {
        "status": "CLASSIFIED",
        "normalized_query": "What is 20% of 500? A. 50 B. 100 C. 150 D. 200",
        "classification": {
            "intent": "solve_question",
            "subject": "math",
            "topic": "percentage",
            "topic_confidence": 0.95,
            "pattern_topic_candidate": "PERCENTAGE",
            "pattern_family_candidate": "ARITHMETIC",
            "retrieval_tags": ["percentage"],
            "response_style": "step_by_step",
            "confidence": 0.96,
            "difficulty": "intermediate",
            "retrieval_need": "concept_context",
            "reasoning_summary": "Quantitative aptitude percentage question.",
            "need_web_search": False,
            "web_search_reason": "",
            "web_search_query": "",
        },
        "image_parse_metadata": {
            "has_question": True,
            "has_options": True,
            "has_visual": False,
            "visual_context": {
                "type": "none",
                "labels": [],
                "entities": [],
                "relationships": [],
                "values": [],
                "description": "",
                "confidence": 0.98,
            },
            "extraction_confidence": 0.98,
            "classification_confidence": 0.96,
            "ignored_content_detected": False,
            "warnings": [],
        },
    }


class FakeProvider:
    def __init__(
        self,
        outcomes: list[object] | None = None,
        *,
        delay_seconds: float = 0.0,
    ) -> None:
        self.outcomes = outcomes or [_provider_output()]
        self.delay_seconds = delay_seconds
        self.call_count = 0
        self.last_request = None

    def classify(self, request):
        self.call_count += 1
        self.last_request = request
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        outcome = self.outcomes[min(self.call_count - 1, len(self.outcomes) - 1)]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_image_provider_usage_is_recorded_once_for_successful_call() -> None:
    provider = FakeProvider(
        [
            _provider_output(
                provider_usage=ProviderTokenUsage(
                    input_tokens=100,
                    output_tokens=20,
                    total_tokens=120,
                )
            )
        ]
    )
    with bind_request_context(request_id="request-image-usage"):
        token = begin_llm_usage_collection()
        try:
            result = _build_classifier(provider).classify(
                image=_image_input(),
                instruction=None,
                request_id="request-image-usage",
            )
            records = snapshot_llm_usage_records()
        finally:
            reset_llm_usage_collection(token)

    assert result.status == ImageClassificationStatus.CLASSIFIED
    assert len(records) == 1
    assert records[0].role == "image_question_classifier"
    assert records[0].provider == "gemini"
    assert records[0].input_tokens == 100
    assert records[0].total_tokens == 120


def test_image_provider_failure_keeps_reported_usage() -> None:
    provider = FakeProvider(
        [
            ImageProviderResponseError(
                "invalid structured response",
                provider_usage=ProviderTokenUsage(
                    input_tokens=80,
                    output_tokens=4,
                    total_tokens=84,
                ),
            )
        ]
    )
    with bind_request_context(request_id="request-image-failure"):
        token = begin_llm_usage_collection()
        try:
            result = _build_classifier(provider).classify(
                image=_image_input(),
                instruction=None,
                request_id="request-image-failure",
            )
            records = snapshot_llm_usage_records()
        finally:
            reset_llm_usage_collection(token)

    assert (
        result.status
        == ImageClassificationStatus.PROVIDER_TEMPORARILY_UNAVAILABLE
    )
    assert len(records) == 1
    assert records[0].status == "failed"
    assert records[0].total_tokens == 84


def _build_classifier(
    provider: FakeProvider,
    *,
    cache: ImageClassificationCache | None = None,
    model: str = "gemini-3.1-flash-lite",
    min_confidence: float = 0.65,
    max_image_bytes: int = 500_000,
    max_dimension: int = 2400,
) -> ImageQuestionClassifier:
    return ImageQuestionClassifier(
        provider=provider,
        validator=ImageInputValidator(
            loader=InlineImageLoader(),
            config=ImageValidationConfig(
                max_image_bytes=max_image_bytes,
                max_dimension=max_dimension,
            ),
        ),
        cache=cache or ImageClassificationCache(ttl_seconds=300),
        provider_name="gemini",
        model=model,
        timeout_seconds=1.0,
        max_output_tokens=1400,
        min_confidence=min_confidence,
    )


class TestRequestContract:
    def test_text_only_request_is_unchanged(self) -> None:
        request = DoubtSolverRequest(
            mode="doubt_solver",
            query="Explain ratio",
            user_id="local-user",
            conversation_id="conversation-1",
            turn_id="turn-1",
        )
        assert request.query == "Explain ratio"
        assert request.image is None

    def test_image_only_request_is_valid(self) -> None:
        request = DoubtSolverRequest(
            mode="doubt_solver",
            image=_image_input(),
            user_id="local-user",
            conversation_id="conversation-1",
            turn_id="turn-1",
        )
        assert request.query is None
        assert request.image is not None

    def test_missing_query_and_image_is_invalid(self) -> None:
        with pytest.raises(ValidationError, match="Either query or image"):
            DoubtSolverRequest(
                mode="doubt_solver",
                user_id="local-user",
                conversation_id="conversation-1",
                turn_id="turn-1",
            )

    def test_image_requires_exactly_one_payload_source(self) -> None:
        with pytest.raises(ValidationError, match="exactly one"):
            ImageInput(
                source="upload",
                mimeType="image/png",
                base64="AAAA",
                storageKey="questions/a.png",
            )


class TestImageValidation:
    def test_corrupt_image_is_rejected_without_provider_call(self) -> None:
        provider = FakeProvider()
        result = _build_classifier(provider).classify(
            image=_image_input(b"not-an-image"), instruction=None, request_id="r1"
        )
        assert result.status == ImageClassificationStatus.REJECTED_UNSUPPORTED_IMAGE
        assert provider.call_count == 0

    def test_mime_signature_mismatch_is_rejected(self) -> None:
        provider = FakeProvider()
        result = _build_classifier(provider).classify(
            image=_image_input(_image_bytes(image_format="JPEG"), mime_type="image/png"),
            instruction=None,
            request_id="r2",
        )
        assert result.status == ImageClassificationStatus.REJECTED_UNSUPPORTED_IMAGE
        assert provider.call_count == 0

    def test_oversized_image_is_rejected_without_retry(self) -> None:
        provider = FakeProvider()
        result = _build_classifier(provider, max_image_bytes=20).classify(
            image=_image_input(), instruction=None, request_id="r3"
        )
        assert result.status == ImageClassificationStatus.REJECTED_UNSUPPORTED_IMAGE
        assert provider.call_count == 0

    def test_unsupported_mime_is_rejected(self) -> None:
        provider = FakeProvider()
        result = _build_classifier(provider).classify(
            image=_image_input(mime_type="image/gif"), instruction=None, request_id="r4"
        )
        assert result.status == ImageClassificationStatus.REJECTED_UNSUPPORTED_IMAGE
        assert provider.call_count == 0

    def test_multiframe_image_is_rejected(self) -> None:
        provider = FakeProvider()
        result = _build_classifier(provider).classify(
            image=_image_input(_animated_webp_bytes(), mime_type="image/webp"),
            instruction=None,
            request_id="r4-multiframe",
        )
        assert result.status == ImageClassificationStatus.REJECTED_UNSUPPORTED_IMAGE
        assert provider.call_count == 0

    def test_small_dimensions_are_rejected(self) -> None:
        provider = FakeProvider()
        result = _build_classifier(provider).classify(
            image=_image_input(_image_bytes(size=(32, 32))),
            instruction=None,
            request_id="r5",
        )
        assert result.status == ImageClassificationStatus.REJECTED_UNSUPPORTED_IMAGE

    def test_large_image_is_resized_preserving_aspect_ratio(self) -> None:
        provider = FakeProvider()
        classifier = _build_classifier(provider, max_dimension=400)
        result = classifier.classify(
            image=_image_input(_image_bytes(size=(1200, 600))),
            instruction=None,
            request_id="r6",
        )
        assert result.status == ImageClassificationStatus.CLASSIFIED
        assert provider.last_request.image.was_resized is True
        assert provider.last_request.image.width == 400
        assert provider.last_request.image.height == 200

    def test_storage_reference_is_not_fetched_by_default(self) -> None:
        provider = FakeProvider()
        image = ImageInput(
            source="upload",
            mimeType="image/png",
            signedUrl="https://example.invalid/private.png?token=secret",
        )
        result = _build_classifier(provider).classify(
            image=image, instruction=None, request_id="r7"
        )
        assert result.status == ImageClassificationStatus.REJECTED_UNSUPPORTED_IMAGE
        assert provider.call_count == 0


class TestClassifierBehavior:
    def test_valid_result_reuses_existing_classification_contract(self) -> None:
        provider = FakeProvider()
        result = _build_classifier(provider).classify(
            image=_image_input(), instruction="Solve question 12 only", request_id="r8"
        )
        assert result.status == ImageClassificationStatus.CLASSIFIED
        assert result.classification is not None
        assert result.classification.model_dump().keys() == _classification().model_dump().keys()
        assert result.classification.classification_source == "llm"
        assert provider.call_count == 1
        assert provider.last_request.instruction == "Solve question 12 only"

    def test_unknown_subject_maps_to_general_only_for_valid_question(self) -> None:
        provider = FakeProvider([_provider_output(subject="unknown")])
        result = _build_classifier(provider).classify(
            image=_image_input(), instruction=None, request_id="r9"
        )
        assert result.classification is not None
        assert result.classification.subject == "general"

    @pytest.mark.parametrize(
        ("fixture_name", "subject", "visual_type"),
        [
            ("quantitative_mcq", "math", "none"),
            ("reasoning_figure", "reasoning", "reasoning_figure"),
            ("english_question", "english", "none"),
            ("general_awareness", "general", "none"),
            ("hindi_bilingual", "math", "none"),
            ("geometry_diagram", "math", "geometry"),
            ("data_interpretation_table", "math", "table"),
            ("screenshot_with_ui", "math", "none"),
        ],
    )
    def test_controlled_question_fixtures_preserve_contract(
        self,
        fixture_name: str,
        subject: str,
        visual_type: str,
    ) -> None:
        provider = FakeProvider(
            [_provider_output(subject=subject, visual_type=visual_type)]
        )
        result = _build_classifier(provider).classify(
            image=_image_input(), instruction=fixture_name, request_id=fixture_name
        )
        assert result.status == ImageClassificationStatus.CLASSIFIED
        assert result.classification is not None
        assert result.classification.subject == subject
        assert result.image_parse_metadata.visual_context.type == visual_type

    def test_confidence_below_threshold_is_rejected(self) -> None:
        provider = FakeProvider([_provider_output(confidence=0.49)])
        result = _build_classifier(provider, min_confidence=0.5).classify(
            image=_image_input(), instruction=None, request_id="r10"
        )
        assert result.status == ImageClassificationStatus.REJECTED_AMBIGUOUS_QUESTION
        assert result.classification is None

    @pytest.mark.parametrize(
        "status",
        [
            ImageClassificationStatus.REJECTED_UNREADABLE_IMAGE,
            ImageClassificationStatus.REJECTED_NO_QUESTION_FOUND,
            ImageClassificationStatus.REJECTED_AMBIGUOUS_QUESTION,
            ImageClassificationStatus.REJECTED_INCOMPLETE_CONTEXT,
        ],
    )
    def test_provider_rejection_states_are_preserved(
        self, status: ImageClassificationStatus
    ) -> None:
        provider = FakeProvider([ImageProviderOutput(status=status)])
        result = _build_classifier(provider).classify(
            image=_image_input(), instruction=None, request_id="r11"
        )
        assert result.status == status
        assert result.user_message

    @pytest.mark.parametrize(
        ("fixture_name", "status"),
        [
            ("two_questions", ImageClassificationStatus.REJECTED_AMBIGUOUS_QUESTION),
            ("blurred_image", ImageClassificationStatus.REJECTED_UNREADABLE_IMAGE),
            ("cropped_question", ImageClassificationStatus.REJECTED_INCOMPLETE_CONTEXT),
            ("non_question", ImageClassificationStatus.REJECTED_NO_QUESTION_FOUND),
        ],
    )
    def test_controlled_rejection_fixtures_fail_closed(
        self,
        fixture_name: str,
        status: ImageClassificationStatus,
    ) -> None:
        provider = FakeProvider([ImageProviderOutput(status=status)])
        result = _build_classifier(provider).classify(
            image=_image_input(), instruction=fixture_name, request_id=fixture_name
        )
        assert result.status == status
        assert result.classification is None

    def test_retryable_failure_retries_once(self) -> None:
        provider = FakeProvider(
            [ImageProviderTemporaryError("timeout"), _provider_output()]
        )
        result = _build_classifier(provider).classify(
            image=_image_input(), instruction=None, request_id="r12"
        )
        assert result.status == ImageClassificationStatus.CLASSIFIED
        assert provider.call_count == 2

    def test_retryable_failure_stops_after_one_retry(self) -> None:
        provider = FakeProvider([ImageProviderTemporaryError("rate limited")])
        result = _build_classifier(provider).classify(
            image=_image_input(), instruction=None, request_id="r13"
        )
        assert result.status == ImageClassificationStatus.PROVIDER_TEMPORARILY_UNAVAILABLE
        assert provider.call_count == 2

    def test_invalid_provider_output_is_not_retried(self) -> None:
        provider = FakeProvider([ImageProviderResponseError("invalid json")])
        result = _build_classifier(provider).classify(
            image=_image_input(), instruction=None, request_id="r14"
        )
        assert result.status == ImageClassificationStatus.PROVIDER_TEMPORARILY_UNAVAILABLE
        assert provider.call_count == 1

    def test_cache_hit_avoids_second_provider_call(self) -> None:
        provider = FakeProvider()
        classifier = _build_classifier(provider)
        first = classifier.classify(image=_image_input(), instruction=None, request_id="r15")
        second = classifier.classify(image=_image_input(), instruction=None, request_id="r16")
        assert first.cache_hit is False
        assert second.cache_hit is True
        assert provider.call_count == 1

    def test_cache_key_includes_instruction(self) -> None:
        provider = FakeProvider()
        classifier = _build_classifier(provider)
        classifier.classify(image=_image_input(), instruction="question 1", request_id="r17")
        classifier.classify(image=_image_input(), instruction="question 2", request_id="r18")
        assert provider.call_count == 2

    def test_cache_key_includes_model_version(self) -> None:
        cache = ImageClassificationCache(ttl_seconds=300)
        first_provider = FakeProvider()
        second_provider = FakeProvider()
        _build_classifier(first_provider, cache=cache, model="model-v1").classify(
            image=_image_input(), instruction=None, request_id="r19"
        )
        _build_classifier(second_provider, cache=cache, model="model-v2").classify(
            image=_image_input(), instruction=None, request_id="r20"
        )
        assert first_provider.call_count == 1
        assert second_provider.call_count == 1

    def test_cache_key_invalidates_when_prompt_changes(self, monkeypatch) -> None:
        cache = ImageClassificationCache(ttl_seconds=300)
        first_provider = FakeProvider()
        second_provider = FakeProvider()
        monkeypatch.setattr(
            classifier_module,
            "build_image_classifier_prompt",
            lambda: "prompt-version-one",
        )
        _build_classifier(first_provider, cache=cache).classify(
            image=_image_input(), instruction=None, request_id="prompt-v1"
        )
        monkeypatch.setattr(
            classifier_module,
            "build_image_classifier_prompt",
            lambda: "prompt-version-two",
        )
        _build_classifier(second_provider, cache=cache).classify(
            image=_image_input(), instruction=None, request_id="prompt-v2"
        )
        assert first_provider.call_count == 1
        assert second_provider.call_count == 1

    def test_concurrent_duplicate_requests_make_one_provider_call(self) -> None:
        provider = FakeProvider(delay_seconds=0.05)
        classifier = _build_classifier(
            provider,
            cache=ImageClassificationCache(ttl_seconds=0),
        )
        image = _image_input()
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(
                executor.map(
                    lambda index: classifier.classify(
                        image=image,
                        instruction=None,
                        request_id=f"concurrent-{index}",
                    ),
                    range(4),
                )
            )
        assert all(result.status == ImageClassificationStatus.CLASSIFIED for result in results)
        assert provider.call_count == 1

    def test_logs_do_not_include_private_payload(self, caplog) -> None:
        encoded = base64.b64encode(_image_bytes()).decode("ascii")
        with caplog.at_level("INFO"):
            _build_classifier(FakeProvider()).classify(
                image=ImageInput(source="upload", mimeType="image/png", base64=encoded),
                instruction="private student question",
                request_id="safe-log",
            )
        assert encoded not in caplog.text
        assert "private student question" not in caplog.text
        assert "outcome_class=success" in caplog.text
        assert "provider_call_count=1" in caplog.text
        assert "text_classifier_bypassed=true" in caplog.text
        assert "metric_count=1" in caplog.text
        assert "provider_latency_ms=" in caplog.text


class TestProviderAndNodeContracts:
    def test_prompt_uses_live_classifier_taxonomy_and_forbids_solving(self) -> None:
        prompt = build_image_classifier_prompt()
        for intent in QueryClassification.model_fields["intent"].annotation.__args__:
            assert intent in prompt
        assert "must not solve" in prompt.lower()
        assert "current office holders" in prompt.lower()
        assert "freshness_required" in prompt
        assert "do not request web search merely because" in prompt.lower()

    def test_gemini_adapter_uses_one_multimodal_structured_call(self) -> None:
        captured = {}

        class Models:
            def generate_content(self, **kwargs):
                captured.update(kwargs)
                return SimpleNamespace(
                    parsed=kwargs["config"].response_schema.model_validate(
                        _gemini_wire_payload()
                    ),
                    text=None,
                    usage_metadata=SimpleNamespace(
                        prompt_token_count=90,
                        candidates_token_count=10,
                        total_token_count=100,
                    ),
                )

        client = SimpleNamespace(models=Models())
        provider = GeminiImageQuestionClassificationProvider(
            api_key="test-key",
            client_factory=lambda **kwargs: client,
        )
        fake = FakeProvider()
        classifier = _build_classifier(fake)
        classifier.classify(image=_image_input(), instruction=None, request_id="seed")
        output = provider.classify(fake.last_request)
        assert output.status == ImageClassificationStatus.CLASSIFIED
        assert output.classification is not None
        assert output.classification.model_dump().keys() == _classification().model_dump().keys()
        assert output.classification.subject == "math"
        assert output.classification.classification_source == "llm"
        assert output.provider_usage is not None
        assert output.provider_usage.input_tokens == 90
        assert output.provider_usage.total_tokens == 100
        assert captured["model"] == "gemini-3.1-flash-lite"
        assert captured["config"].response_schema is not ImageProviderOutput
        assert len(captured["contents"]) == 2

        wire_schema = captured["config"].response_schema.model_json_schema()
        encoded_schema = json.dumps(wire_schema)
        assert '"anyOf"' not in encoded_schema
        assert '"additionalProperties": true' not in encoded_schema

    def test_gemini_adapter_maps_current_information_search_demand(self) -> None:
        payload = _gemini_wire_payload()
        payload["normalized_query"] = "Who is the current holder of this office?"
        payload["classification"].update(
            {
                "subject": "general",
                "topic": "current office holder",
                "pattern_topic_candidate": "",
                "pattern_family_candidate": "",
                "retrieval_tags": ["current_office_holder"],
                "retrieval_need": "none",
                "need_web_search": True,
                "web_search_reason": "freshness_required",
                "web_search_query": "current holder of the named office official",
            }
        )

        class Models:
            def generate_content(self, **kwargs):
                return SimpleNamespace(
                    parsed=kwargs["config"].response_schema.model_validate(payload),
                    text=None,
                    usage_metadata=None,
                )

        client = SimpleNamespace(models=Models())
        provider = GeminiImageQuestionClassificationProvider(
            api_key="test-key",
            client_factory=lambda **kwargs: client,
        )
        seed_provider = FakeProvider()
        classifier = _build_classifier(seed_provider)
        classifier.classify(image=_image_input(), instruction=None, request_id="seed-current")

        output = provider.classify(seed_provider.last_request)

        assert output.classification is not None
        assert output.classification.need_web_search is True
        assert output.classification.web_search_reason == "freshness_required"
        assert (
            output.classification.web_search_query
            == "current holder of the named office official"
        )

    @pytest.mark.parametrize(
        "content",
        ["not-json", json.dumps({"status": "CLASSIFIED", "normalized_query": "x"})],
    )
    def test_gemini_adapter_rejects_invalid_or_incomplete_json(self, content: str) -> None:
        class Models:
            def generate_content(self, **kwargs):  # noqa: ARG002
                return SimpleNamespace(parsed=None, text=content, usage_metadata=None)

        client = SimpleNamespace(models=Models())
        provider = GeminiImageQuestionClassificationProvider(
            api_key="test-key", client_factory=lambda **kwargs: client
        )
        fake = FakeProvider()
        classifier = _build_classifier(fake)
        classifier.classify(image=_image_input(), instruction=None, request_id="seed2")
        with pytest.raises(ImageProviderResponseError):
            provider.classify(fake.last_request)

    def test_graph_node_depends_only_on_classifier_protocol(self) -> None:
        expected = ImageClassificationResult(
            status=ImageClassificationStatus.CLASSIFIED,
            normalized_query="Question",
            classification=_classification(),
            image_parse_metadata=_metadata(),
        )

        class FakeClassifier:
            def classify(self, **kwargs):
                assert kwargs["request_id"] == "node-1"
                return expected

        result = image_classifier_node(
            ImageClassifierNodeInput(request_id="node-1", image=_image_input()),
            classifier=FakeClassifier(),
        )
        assert result == expected


class TestConfiguration:
    def test_enabled_without_api_key_fails_fast(self, monkeypatch) -> None:
        monkeypatch.setenv("IMAGE_CLASSIFIER_ENABLED", "true")
        monkeypatch.setenv("GOOGLE_GEMINI_API_KEY", "")
        config_module._settings = None
        with pytest.raises(config_module.ConfigurationError, match="GOOGLE_GEMINI_API_KEY"):
            config_module.get_settings()

    def test_disabled_does_not_require_api_key(self, monkeypatch) -> None:
        monkeypatch.setenv("IMAGE_CLASSIFIER_ENABLED", "false")
        monkeypatch.setenv("GOOGLE_GEMINI_API_KEY", "")
        config_module._settings = None
        assert config_module.get_settings().image_classifier_enabled is False

    def test_disabled_ignores_invalid_image_numeric_settings(self, monkeypatch) -> None:
        monkeypatch.setenv("IMAGE_CLASSIFIER_ENABLED", "false")
        monkeypatch.setenv("IMAGE_CLASSIFIER_TIMEOUT_MS", "not-a-number")
        config_module._settings = None
        assert config_module.get_settings().image_classifier_timeout_ms == 15000

    def test_default_model_is_stable_flash_lite(self, monkeypatch) -> None:
        monkeypatch.delenv("IMAGE_CLASSIFIER_MODEL", raising=False)
        config_module._settings = None
        assert config_module.get_settings().image_classifier_model == "gemini-3.1-flash-lite"

    def test_enabled_with_unsupported_provider_fails_fast(self, monkeypatch) -> None:
        monkeypatch.setenv("IMAGE_CLASSIFIER_ENABLED", "true")
        monkeypatch.setenv("IMAGE_CLASSIFIER_PROVIDER", "unsupported")
        monkeypatch.setenv("GOOGLE_GEMINI_API_KEY", "configured")
        config_module._settings = None
        with pytest.raises(config_module.ConfigurationError, match="IMAGE_CLASSIFIER_PROVIDER"):
            config_module.get_settings()

    def test_enabled_with_invalid_confidence_fails_fast(self, monkeypatch) -> None:
        monkeypatch.setenv("IMAGE_CLASSIFIER_ENABLED", "true")
        monkeypatch.setenv("IMAGE_CLASSIFIER_PROVIDER", "gemini")
        monkeypatch.setenv("GOOGLE_GEMINI_API_KEY", "configured")
        monkeypatch.setenv("IMAGE_CLASSIFIER_MIN_CONFIDENCE", "1.5")
        config_module._settings = None
        with pytest.raises(config_module.ConfigurationError, match="MIN_CONFIDENCE"):
            config_module.get_settings()
