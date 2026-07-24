"""Isolated entry-routing tests for image and text doubt-solver requests."""

from __future__ import annotations

import base64
import io

from PIL import Image

import graphs.doubt_solver_graph as doubt_solver_graph_module
import main
import services.doubt_solver.streaming_doubt_solver_service as streaming_module
from schemas.doubt_solver import QueryClassification
from schemas.image_question_classification import (
    ImageClassificationResult,
    ImageClassificationStatus,
    ImageParseMetadata,
)

_REQUEST_IDS = {
    "user_id": "local-user",
    "conversation_id": "conversation-image-entry",
    "turn_id": "turn-image-entry",
}


def _encoded_image() -> str:
    output = io.BytesIO()
    Image.new("RGB", (320, 180), "white").save(output, format="PNG")
    return base64.b64encode(output.getvalue()).decode("ascii")


def _image_result() -> ImageClassificationResult:
    return ImageClassificationResult(
        status=ImageClassificationStatus.CLASSIFIED,
        normalized_query="What is 20% of 500? A. 50 B. 100 C. 150 D. 200",
        classification=QueryClassification(
            intent="solve_question",
            subject="math",
            topic="percentage",
            confidence=0.96,
            difficulty="intermediate",
            retrieval_need="concept_context",
            classification_source="llm",
        ),
        image_parse_metadata=ImageParseMetadata(
            has_question=True,
            extraction_confidence=0.97,
            classification_confidence=0.96,
        ),
    )


class FakeImageClassifier:
    def __init__(self, result: ImageClassificationResult) -> None:
        self.result = result
        self.call_count = 0

    def classify(self, **kwargs):
        self.call_count += 1
        return self.result


def test_disabled_image_request_returns_controlled_unsupported_state(monkeypatch) -> None:
    monkeypatch.setattr(main, "image_question_classifier", None)
    result = main.invoke(
        {
            "mode": "doubt_solver",
            **_REQUEST_IDS,
            "image": {
                "source": "upload",
                "mimeType": "image/png",
                "base64": _encoded_image(),
            },
        }
    )
    assert result["success"] is False
    assert result["image_classification_status"] == "REJECTED_UNSUPPORTED_IMAGE"
    assert "Gemini" not in result["answer"]


def test_image_classification_flows_into_existing_downstream_graph(monkeypatch) -> None:
    fake = FakeImageClassifier(_image_result())
    monkeypatch.setattr(main, "image_question_classifier", fake)

    def fail_if_text_classifier_runs(*args, **kwargs):
        raise AssertionError("text classifier must not run for an image request")

    monkeypatch.setattr("graphs.doubt_solver_graph.classify_query", fail_if_text_classifier_runs)
    result = main.invoke(
        {
            "mode": "doubt_solver",
            **_REQUEST_IDS,
            "query": "Solve this question only",
            "image": {
                "source": "camera",
                "mimeType": "image/png",
                "base64": _encoded_image(),
            },
        }
    )
    assert result["success"] is True
    assert result["classification"]["subject"] == "math"
    assert result["classification"]["classification_source"] == "llm"
    assert fake.call_count == 1
    assert "image_classification_status" not in result
    assert "image_parse_metadata" not in result


def test_text_request_does_not_call_image_classifier(monkeypatch) -> None:
    class FailingImageClassifier:
        def classify(self, **kwargs):
            raise AssertionError("image classifier must not run for text requests")

    monkeypatch.setattr(main, "image_question_classifier", FailingImageClassifier())
    result = main.invoke(
        {"mode": "doubt_solver", "query": "Explain ratio", **_REQUEST_IDS}
    )
    assert result["success"] is True
    assert result["classification"] is not None


def test_image_rejection_never_reaches_downstream_graph(monkeypatch) -> None:
    fake = FakeImageClassifier(
        ImageClassificationResult(
            status=ImageClassificationStatus.REJECTED_AMBIGUOUS_QUESTION,
            user_message=(
                "Multiple questions were detected. Please crop and upload the question you want."
            ),
        )
    )
    monkeypatch.setattr(main, "image_question_classifier", fake)

    def fail_if_graph_runs(*args, **kwargs):
        raise AssertionError("downstream graph must not run for rejected images")

    monkeypatch.setattr(main.doubt_solver_graph, "invoke", fail_if_graph_runs)
    result = main.invoke(
        {
            "mode": "doubt_solver",
            **_REQUEST_IDS,
            "image": {
                "source": "screenshot",
                "mimeType": "image/png",
                "base64": _encoded_image(),
            },
        }
    )
    assert result["success"] is False
    assert result["image_classification_status"] == "REJECTED_AMBIGUOUS_QUESTION"


def test_orchestrated_node_reuses_image_classification(monkeypatch) -> None:
    expected = {
        "subject": "math",
        "intent": "solve",
        "difficulty": "intermediate",
        "retrieval_required": False,
    }

    def fail_if_text_classifier_runs(*args, **kwargs):
        raise AssertionError("text classifier must not run")

    monkeypatch.setattr(
        doubt_solver_graph_module,
        "orchestrated_classify_query",
        fail_if_text_classifier_runs,
    )
    result = doubt_solver_graph_module._orchestrated_classify_node(
        {
            "request_id": "image-orchestrated",
            "query": "Extracted image question",
            "classification": expected,
            "retrieval_context": {},
            "context_text": "",
            "answer": None,
        }
    )
    assert result == {"classification": expected}


def test_streaming_reuses_image_classification(monkeypatch) -> None:
    expected = {
        "subject": "math",
        "intent": "solve",
        "difficulty": "intermediate",
        "retrieval_required": False,
        "need_web_search": False,
    }

    def fail_if_text_classifier_runs(*args, **kwargs):
        raise AssertionError("text classifier must not run")

    monkeypatch.setattr(
        streaming_module,
        "orchestrated_classify_query_with_delivery_signals",
        fail_if_text_classifier_runs,
    )
    monkeypatch.setattr(
        streaming_module,
        "_orchestrated_collect_context_node",
        lambda state, **kwargs: {"context_text": "", "retrieval_context": {}},
    )

    class FakeStreamingAdapter:
        def generate(self, **kwargs):
            assert kwargs["query"] == "Extracted image question"
            return "**Final Answer:**\n\\(20\\)"

        def generate_stream(self, **kwargs):
            assert kwargs["query"] == "Extracted image question"
            yield "**Final Answer:**\n\\(20\\)"

    events = list(
        streaming_module.stream_doubt_solver(
            streaming_module.StreamDoubtSolverInput(
                request_id="image-stream",
                query="Extracted image question",
                classification=expected,
            ),
            adapter=FakeStreamingAdapter(),
        )
    )
    assert "".join(event.content or "" for event in events if event.type == "chunk") == (
        "**Final Answer:**\n\\(20\\)"
    )
