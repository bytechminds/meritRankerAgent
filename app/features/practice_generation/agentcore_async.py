"""AgentCore-tracked background execution for the existing practice graph."""

from __future__ import annotations

import json
import logging
import re
import threading
from concurrent.futures import Executor, ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any, Protocol

from config import get_settings
from features.practice_generation.appsync_progress_client import AppSyncPracticeProgressClient
from features.practice_generation.config import (
    PRACTICE_RUNTIME_REVISION,
    PracticeConfigurationError,
    get_practice_config,
)
from features.practice_generation.events import emit_practice_event, safe_user_ref
from features.practice_generation.graph import PracticeGraphRunner
from features.practice_generation.orchestration import PracticeGenerationOrchestrator
from features.practice_generation.pattern_context import build_pattern_context_provider
from features.practice_generation.planning import (
    BlueprintManager,
    deterministic_test_id,
    request_idempotency_key,
)
from features.practice_generation.progress import AppSyncAssessmentProgressRepository
from features.practice_generation.providers import (
    RoutedPlannerProvider,
    RoutedQuestionGenerator,
    RoutedQuestionVerifier,
)
from features.practice_generation.repositories import (
    AssessmentRepository,
    PracticeRepositoryError,
    QuestionRepository,
)
from features.practice_generation.resource_validation import validate_practice_resources
from features.practice_generation.schemas import (
    PracticeGenerationRequest,
    PracticeGraphCommand,
    PracticeLaunchResult,
)
from observability import (
    bind_execution_context,
    bind_request_context,
    current_request_context,
)
from services.aws_client_factory import get_dynamodb_client
from services.llm.billing import (
    OperationUsageAccumulator,
    begin_operation,
    emit_operation_billing_summary,
    feature_for_practice_type,
    hand_off_current_operation,
)
from services.llm.orchestration.orchestrator import LlmOrchestrator

logger = logging.getLogger(__name__)

_MAX_BACKGROUND_TASKS = 4
_MAX_GRAPH_OPERATIONS = 500
_SAFE_FAILURE_CODE = re.compile(r"^[A-Z][A-Z0-9_]{2,95}$")


class AgentCoreTaskTracker(Protocol):
    """Public AgentCore task-tracking methods used by this integration."""

    def add_async_task(self, name: str, metadata: dict | None = None) -> int: ...

    def complete_async_task(self, task_id: int) -> bool: ...


class PracticeLaunchError(RuntimeError):
    """Typed failure raised before an acknowledgement can be returned."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _failure_reason_code(exc: BaseException) -> str:
    """Expose only controlled repository codes at the retained-task boundary."""
    if isinstance(exc, PracticeRepositoryError):
        code = str(exc)
        if _SAFE_FAILURE_CODE.fullmatch(code):
            return code
    return "PRACTICE_GENERATION_FAILED"


def _safe_failure_details(exc: BaseException) -> dict[str, str]:
    """Return correlation and causal metadata without exposing user content."""
    details: dict[str, str] = {}
    request_context = current_request_context()
    if request_context is not None:
        details["requestId"] = request_context.request_id
    if isinstance(exc, PracticeRepositoryError):
        details.update(exc.safe_details())
    cause = exc.__cause__
    if cause is not None:
        details["causeClass"] = type(cause).__name__
        response = getattr(cause, "response", None)
        error = response.get("Error") if isinstance(response, dict) else None
        if isinstance(error, dict) and error.get("Code"):
            details.setdefault("awsErrorCode", str(error["Code"]))
    return details


def _meta(assessment: dict[str, Any]) -> dict[str, Any]:
    value = assessment.get("meta")
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _message(language: str, status: str) -> str:
    if status == "FAILED":
        return "Practice generation could not be completed. Please try again."
    if status == "READY":
        return "Your practice test is ready."
    if language == "hindi":
        return "आपका प्रैक्टिस टेस्ट तैयार किया जा रहा है।"
    if language == "hinglish":
        return "Aapka practice test taiyar kiya ja raha hai."
    return "Your practice test is being prepared."


class AgentCorePracticeAsyncLauncher:
    """Initialize an assessment and run its retained graph as one tracked task."""

    def __init__(
        self,
        *,
        task_tracker: AgentCoreTaskTracker,
        assessments: AssessmentRepository,
        progress: AppSyncAssessmentProgressRepository,
        graph_runner: PracticeGraphRunner,
        executor: Executor | None = None,
        recovery_stale_seconds: int = 120,
        max_wall_time_seconds: int = 900,
    ) -> None:
        self._task_tracker = task_tracker
        self._assessments = assessments
        self._progress = progress
        self._graph_runner = graph_runner
        self._recovery_stale_seconds = min(max(recovery_stale_seconds, 30), 3_600)
        self._max_wall_time_seconds = min(max(max_wall_time_seconds, 60), 3_600)
        self._executor = executor or ThreadPoolExecutor(
            max_workers=_MAX_BACKGROUND_TASKS,
            thread_name_prefix="practice-agentcore-task",
        )
        self._active_test_ids: set[str] = set()
        self._pending_task_ids: dict[str, int] = {}
        self._pending_delivery_ids: dict[str, set[str]] = {}
        self._closing_test_ids: set[str] = set()
        self._billing_operations: dict[str, OperationUsageAccumulator] = {}
        self._active_lock = threading.Lock()

    def launch(self, request: PracticeGenerationRequest) -> PracticeLaunchResult:
        test_id = deterministic_test_id(request_idempotency_key(request))
        try:
            assessment, duplicate = self._assessments.create_or_get(test_id, request)
        except Exception as exc:
            raise PracticeLaunchError("PRACTICE_INITIALIZATION_FAILED") from exc

        emit_practice_event(
            "practice_assessment_initialized",
            test_id=test_id,
            status=str(assessment.get("status") or "GENERATING").casefold(),
            details={
                "actorRef": safe_user_ref(request.user_id),
                "requestedCount": request.requested_count,
                "acceptedCount": request.accepted_count,
                "duplicateRequest": duplicate,
            },
        )
        status = str(assessment.get("status") or "GENERATING")
        if status not in {"GENERATING", "READY", "FAILED"}:
            status = "FAILED"
        if status == "GENERATING":
            if not duplicate:
                self._register(test_id, request.request_id)
            elif self._is_stale(assessment):
                recovery_count = int(_meta(assessment).get("recoveryAttemptCount") or 0)
                if recovery_count >= 1:
                    self._mark_failed(test_id, "PRACTICE_RECOVERY_EXHAUSTED")
                    assessment = self._assessments.get(test_id) or assessment
                    status = str(assessment.get("status") or "FAILED")
                elif self._progress.claim_recovery(test_id):
                    self._register(test_id, request.request_id)
            elif test_id in self._active_test_ids or test_id in self._closing_test_ids:
                self._register(test_id, request.request_id)
        if status == "GENERATING":
            self._attach_billing_operation(test_id, request.practice_type)
        meta = _meta(assessment)
        progress = int(meta.get("progressPercent") or 0)
        return PracticeLaunchResult(
            test_id=test_id,
            status=status,
            requested_count=request.requested_count,
            accepted_count=request.accepted_count,
            count_clamped=request.requested_count != request.accepted_count,
            progress_percent=max(0, min(progress, 100)),
            playable=status == "READY" and bool(meta.get("playable")),
            duplicate_request=duplicate,
            message=_message(request.language, status),
        )

    def _attach_billing_operation(self, test_id: str, practice_type: str) -> None:
        """Move the current request accumulator to the durable practice lifecycle."""
        try:
            feature = feature_for_practice_type(practice_type)
            incoming = hand_off_current_operation(
                operation_id=test_id,
                feature=feature,
            )
            if incoming is None:
                return
            with self._active_lock:
                existing = self._billing_operations.get(test_id)
                if existing is None:
                    self._billing_operations[test_id] = incoming
                elif existing is not incoming:
                    if not existing.absorb(incoming):
                        incoming.clear_handoff()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ai_usage_meter_unavailable reason=practice_handoff_failed error_type=%s",
                type(exc).__name__,
            )

    def _operation_for_task(
        self,
        *,
        test_id: str,
        practice_type: str | None,
    ) -> OperationUsageAccumulator | None:
        with self._active_lock:
            accumulator = self._billing_operations.get(test_id)
        if accumulator is not None:
            accumulator.clear_handoff()
            return accumulator
        try:
            accumulator = begin_operation(
                operation_id=test_id,
                feature=feature_for_practice_type(practice_type),
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "ai_usage_meter_unavailable reason=practice_start_failed error_type=%s",
                type(exc).__name__,
            )
            return None
        if accumulator is not None:
            with self._active_lock:
                self._billing_operations.setdefault(test_id, accumulator)
                accumulator = self._billing_operations[test_id]
        return accumulator

    def _finalize_billing(self, test_id: str, operation_status: str) -> None:
        with self._active_lock:
            accumulator = self._billing_operations.get(test_id)
        if accumulator is None:
            return
        accumulator.clear_handoff()
        with bind_execution_context(
            operation_id=test_id,
            feature=accumulator.descriptor.feature,
            operation_accumulator=accumulator,
        ):
            emit_operation_billing_summary(operation_status=operation_status)

    def _is_stale(self, assessment: dict[str, Any]) -> bool:
        meta = _meta(assessment)
        raw = meta.get("lastProgressAt") or assessment.get("updatedAt")
        if not isinstance(raw, str) or not raw.strip():
            return False
        try:
            timestamp = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            return False
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=UTC)
        return (datetime.now(UTC) - timestamp).total_seconds() >= self._recovery_stale_seconds

    def _register(self, test_id: str, delivery_id: str) -> None:
        registration_error: Exception | None = None
        with self._active_lock:
            if test_id in self._active_test_ids:
                self._pending_delivery_ids.setdefault(test_id, set()).add(delivery_id)
                return
            try:
                current = self._assessments.get(test_id)
            except Exception as exc:
                raise PracticeLaunchError("PRACTICE_INITIALIZATION_FAILED") from exc
            if current is None or str(current.get("status") or "") != "GENERATING":
                raise PracticeLaunchError("PRACTICE_INITIALIZATION_FAILED")
            self._pending_delivery_ids.setdefault(test_id, set()).add(delivery_id)
            self._active_test_ids.add(test_id)
            try:
                task_id = self._task_tracker.add_async_task(
                    "practice_generation",
                    {"testId": test_id},
                )
            except Exception as exc:
                self._active_test_ids.discard(test_id)
                self._pending_delivery_ids.pop(test_id, None)
                self._closing_test_ids.add(test_id)
                try:
                    self._mark_failed(test_id, "PRACTICE_ASYNC_TASK_REGISTRATION_FAILED")
                finally:
                    self._closing_test_ids.discard(test_id)
                registration_error = exc
            else:
                self._pending_task_ids[test_id] = task_id
        if registration_error is not None:
            raise PracticeLaunchError("PRACTICE_INITIALIZATION_FAILED") from registration_error

        emit_practice_event(
            "practice_async_task_registered",
            test_id=test_id,
            status="registered",
        )

    def start(self, test_id: str, delivery_id: str | None = None) -> bool:
        """Start a previously registered task after conversation linkage is durable."""
        with self._active_lock:
            resolved_delivery_id = self._resolve_delivery_id(test_id, delivery_id)
            if resolved_delivery_id is None:
                return False
            task_id = self._pending_task_ids.pop(test_id, None)
            if task_id is None:
                owners = self._pending_delivery_ids.get(test_id)
                if owners is not None:
                    owners.discard(resolved_delivery_id)
                    if not owners:
                        self._pending_delivery_ids.pop(test_id, None)
                return False
            self._pending_delivery_ids.pop(test_id, None)
        try:
            self._executor.submit(self._run_tracked, task_id, test_id)
        except Exception as exc:
            self._mark_failed(test_id, "PRACTICE_ASYNC_TASK_START_FAILED")
            self._finalize_billing(test_id, "failed")
            self._discard_active(test_id)
            self._task_tracker.complete_async_task(task_id)
            raise PracticeLaunchError("PRACTICE_INITIALIZATION_FAILED") from exc
        return True

    def abort(
        self,
        test_id: str,
        code: str,
        delivery_id: str | None = None,
    ) -> None:
        """Close a registered task that cannot be safely linked to its conversation."""
        task_id: int | None = None
        with self._active_lock:
            resolved_delivery_id = self._resolve_delivery_id(test_id, delivery_id)
            if resolved_delivery_id is None:
                return
            owners = self._pending_delivery_ids.get(test_id)
            if owners is None:
                return
            owners.discard(resolved_delivery_id)
            if owners:
                return
            self._pending_delivery_ids.pop(test_id, None)
            task_id = self._pending_task_ids.pop(test_id, None)
            if task_id is None:
                return
            self._active_test_ids.discard(test_id)
            self._closing_test_ids.add(test_id)
            try:
                self._mark_failed(test_id, code)
            finally:
                self._closing_test_ids.discard(test_id)
        if task_id is not None:
            self._finalize_billing(test_id, "failed")
            self._task_tracker.complete_async_task(task_id)
            self._discard_active(test_id)

    def _resolve_delivery_id(
        self,
        test_id: str,
        delivery_id: str | None,
    ) -> str | None:
        owners = self._pending_delivery_ids.get(test_id)
        if not owners:
            return None
        if delivery_id is not None:
            return delivery_id if delivery_id in owners else None
        if len(owners) == 1:
            return next(iter(owners))
        return None

    def _run_tracked(self, task_id: int, test_id: str) -> None:
        """Restore safe request correlation before the background graph starts."""
        request_id = test_id
        conversation_id: str | None = None
        turn_id: str | None = None
        practice_type: str | None = None
        try:
            assessment = self._assessments.get(test_id) or {}
            practice_request = _meta(assessment).get("practiceRequest")
            if isinstance(practice_request, dict):
                request_id = str(practice_request.get("requestId") or test_id)
                conversation_id = str(practice_request.get("conversationId") or "") or None
                turn_id = str(practice_request.get("turnId") or "") or None
                raw_practice_type = practice_request.get("practiceType")
                practice_type = (
                    str(raw_practice_type) if raw_practice_type is not None else None
                )
        except Exception:  # noqa: BLE001
            pass
        with bind_request_context(
            request_id=request_id,
            conversation_id=conversation_id,
            turn_id=turn_id,
            request_type="pending",
        ):
            accumulator = self._operation_for_task(
                test_id=test_id,
                practice_type=practice_type,
            )
            with bind_execution_context(
                activity_id=test_id,
                operation_id=test_id,
                feature=(
                    accumulator.descriptor.feature if accumulator is not None else None
                ),
                operation_accumulator=accumulator,
            ):
                self._run_tracked_with_context(task_id, test_id)

    def _run_tracked_with_context(self, task_id: int, test_id: str) -> None:
        business_status = "FAILED"
        try:
            emit_practice_event(
                "practice_graph_started",
                test_id=test_id,
                status="started",
            )
            business_status = self._run_to_terminal(test_id)
            if business_status not in {"READY", "FAILED"}:
                self._mark_failed(test_id, "PRACTICE_GENERATION_STALLED")
                business_status = "FAILED"
        except BaseException as exc:  # the task boundary must consume background failures
            failure_class = type(exc).__name__
            failure_reason_code = _failure_reason_code(exc)
            failure_details = _safe_failure_details(exc)
            logger.error(
                "practice background task failed test_id=%s error_type=%s operation=%s "
                "cause=%s aws_error_code=%s",
                test_id,
                failure_class,
                failure_details.get("repositoryOperation", "unknown"),
                failure_details.get("causeClass", "none"),
                failure_details.get("awsErrorCode", "none"),
            )
            emit_practice_event(
                "practice_async_task_failed",
                test_id=test_id,
                status="failed",
                details={
                    "reasonCode": failure_reason_code,
                    "errorClass": failure_class,
                    **failure_details,
                },
                level=logging.ERROR,
            )
            self._mark_failed(test_id, failure_reason_code)
            business_status = "FAILED"
        finally:
            emit_operation_billing_summary(operation_status=business_status.casefold())
            self._discard_active(test_id)
            self._task_tracker.complete_async_task(task_id)
            emit_practice_event(
                "practice_async_task_completed",
                test_id=test_id,
                status="completed",
                details=self._terminal_task_details(test_id, business_status),
            )

    def _terminal_task_details(self, test_id: str, business_status: str) -> dict[str, Any]:
        try:
            assessment = self._assessments.get(test_id) or {}
        except Exception:  # noqa: BLE001
            assessment = {}
        meta = _meta(assessment)
        request = meta.get("practiceRequest")
        request_data = request if isinstance(request, dict) else {}
        requested_count = int(
            assessment.get("totalQuestions")
            or request_data.get("acceptedCount")
            or 0
        )
        accepted_count = int(meta.get("readyCount") or 0)
        details: dict[str, Any] = {
            "executionStatus": "completed",
            "businessStatus": business_status,
            "acceptedCount": accepted_count,
            "requestedCount": requested_count,
            "remainingCount": max(requested_count - accepted_count, 0),
        }
        error_code = meta.get("errorCode")
        if isinstance(error_code, str) and error_code:
            details["reasonCode"] = error_code
        return details

    def _run_to_terminal(self, test_id: str) -> str:
        self._graph_runner.process(PracticeGraphCommand(operation="plan_and_fill", test_id=test_id))
        for _operation in range(_MAX_GRAPH_OPERATIONS):
            assessment = self._assessments.get(test_id)
            if assessment is None:
                raise PracticeLaunchError("PRACTICE_PERSISTENCE_FAILED")
            status = str(assessment.get("status") or "")
            if status in {"READY", "FAILED"}:
                return status
            meta = _meta(assessment)
            started_at = meta.get("startedAt")
            if isinstance(started_at, str):
                try:
                    started = datetime.fromisoformat(started_at.replace("Z", "+00:00"))
                    if started.tzinfo is None:
                        started = started.replace(tzinfo=UTC)
                    if (
                        datetime.now(UTC) - started
                    ).total_seconds() >= self._max_wall_time_seconds:
                        self._mark_failed(test_id, "PRACTICE_MAX_WALL_TIME_EXCEEDED")
                        return "FAILED"
                except ValueError:
                    pass
            groups = dict(meta.get("generationGroups") or {})
            pending = [
                (group_id, group)
                for group_id, group in groups.items()
                if isinstance(group, dict) and group.get("state") == "PENDING"
            ]
            if pending:
                blueprint = meta.get("blueprint")
                schema_version = (
                    str(blueprint.get("schema_version") or "1")
                    if isinstance(blueprint, dict)
                    else "1"
                )
                if schema_version == "2":
                    self._graph_runner.process(
                        PracticeGraphCommand(
                            operation="generate_wave",
                            test_id=test_id,
                            group_ids=[group_id for group_id, _group in pending[:2]],
                        )
                    )
                    continue
                group_id, group = pending[0]
                self._graph_runner.process(
                    PracticeGraphCommand(
                        operation="generate_group",
                        test_id=test_id,
                        group_id=group_id,
                        attempt=int(group.get("attempt") or 0),
                    )
                )
                continue
            if any(
                isinstance(group, dict) and group.get("state") == "RUNNING"
                for group in groups.values()
            ):
                return "DELEGATED"
            if groups and all(
                isinstance(group, dict) and group.get("state") == "COMPLETED"
                for group in groups.values()
            ):
                self._graph_runner.process(
                    PracticeGraphCommand(
                        operation="finalize",
                        test_id=test_id,
                        attempt=int(meta.get("finalizationAttempt") or 0),
                    )
                )
                continue
            return status or "GENERATING"
        raise PracticeLaunchError("PRACTICE_GENERATION_OPERATION_LIMIT_EXCEEDED")

    def _mark_failed(self, test_id: str, code: str) -> None:
        try:
            self._progress.mark_failed(test_id, code)
        except Exception as exc:
            logger.error(
                "practice terminal failure persistence failed test_id=%s error_type=%s",
                test_id,
                type(exc).__name__,
            )
        emit_practice_event(
            "practice_failed",
            test_id=test_id,
            status="failed",
            details={"reasonCode": code},
            level=logging.ERROR,
        )
        emit_practice_event(
            "practice_failed_published",
            test_id=test_id,
            status="failed",
            details={"reasonCode": code},
            level=logging.ERROR,
        )
        emit_practice_event(
            "practice_generation_terminal",
            test_id=test_id,
            status="failed",
            details={"reasonCode": code, "businessStatus": "FAILED"},
            level=logging.ERROR,
        )

    def _discard_active(self, test_id: str) -> None:
        with self._active_lock:
            self._active_test_ids.discard(test_id)
            self._pending_task_ids.pop(test_id, None)
            self._pending_delivery_ids.pop(test_id, None)
            self._billing_operations.pop(test_id, None)


def build_practice_async_launcher(
    *,
    task_tracker: AgentCoreTaskTracker,
    llm_orchestrator: LlmOrchestrator,
) -> AgentCorePracticeAsyncLauncher | None:
    """Build the production practice graph from existing runtime integrations."""
    config = get_practice_config()
    if not config.enabled:
        return None
    runtime_settings = get_settings()
    if config.pattern_reuse_enabled and not all(
        (
            runtime_settings.pattern_intelligence_enabled,
            runtime_settings.dynamodb_question_bank_pattern_index,
            runtime_settings.dynamodb_practice_attempt_table,
            runtime_settings.dynamodb_practice_attempt_user_index,
        )
    ):
        raise PracticeConfigurationError(
            "Pattern reuse requires QuestionBank Pattern and practice-history indexes."
        )
    dynamodb = get_dynamodb_client(config.aws_region)
    capabilities = validate_practice_resources(config, dynamodb=dynamodb)
    if capabilities is None:
        return None
    emit_practice_event(
        "PRACTICE_RUNTIME_STARTED",
        test_id="runtime",
        status="completed",
        details={
            "runtimeRevision": PRACTICE_RUNTIME_REVISION,
            "resourceContractVersion": config.resource_schema_version,
            "reuseKeyContractVersion": config.reuse_key_contract_version,
            "patternContextEnabled": config.pattern_context_enabled,
            "patternReuseEnabled": config.pattern_reuse_enabled,
        },
    )
    assessments = AssessmentRepository(
        dynamodb,
        table_name=config.assessment_table,
        meta_safe_size_bytes=config.meta_safe_size_bytes,
    )
    questions = QuestionRepository(
        dynamodb,
        assessment_table=config.assessment_table,
        question_table=config.question_table,
        question_bank_table=config.question_bank_table,
        question_test_index=config.question_test_index,
        question_bank_category_index=config.question_bank_category_index,
        pattern_context_enabled=config.pattern_context_enabled,
        pattern_reuse_enabled=config.pattern_reuse_enabled,
        practice_attempt_table=runtime_settings.dynamodb_practice_attempt_table,
        practice_attempt_user_index=(
            runtime_settings.dynamodb_practice_attempt_user_index
        ),
        question_bank_reuse_index=config.question_bank_reuse_index,
        question_bank_reuse_projection=capabilities.question_bank_reuse,
        question_bank_category_projection=capabilities.question_bank_category,
        query_page_size=config.question_bank_query_page_size,
        query_max_pages=config.question_bank_max_pages,
        query_latency_warning_ms=config.query_latency_warning_ms,
    )
    progress = AppSyncAssessmentProgressRepository(
        assessments=assessments,
        questions=questions,
        client=AppSyncPracticeProgressClient(
            endpoint=config.appsync_graphql_endpoint,
            region=config.aws_region,
        ),
    )
    orchestrator = PracticeGenerationOrchestrator(
        config=config,
        assessments=assessments,
        progress=progress,
        questions=questions,
        blueprint_manager=BlueprintManager(
            RoutedPlannerProvider(llm_orchestrator),
            repair_limit=config.planner_repair_limit,
        ),
        generator=RoutedQuestionGenerator(
            llm_orchestrator,
            pattern_input_max_tokens=(
                runtime_settings.pattern_intelligence_prompt_max_input_tokens
            ),
        ),
        verifier=RoutedQuestionVerifier(llm_orchestrator),
        pattern_context=build_pattern_context_provider(
            enabled=(
                config.pattern_context_enabled or config.pattern_reuse_enabled
            )
        ),
    )
    return AgentCorePracticeAsyncLauncher(
        task_tracker=task_tracker,
        assessments=assessments,
        progress=progress,
        graph_runner=PracticeGraphRunner(orchestrator),
        recovery_stale_seconds=config.recovery_stale_seconds,
        max_wall_time_seconds=config.max_wall_time_seconds,
    )
