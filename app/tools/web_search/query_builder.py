"""Build provider-neutral search requests from policy and attempt kind."""

from __future__ import annotations

from dataclasses import dataclass

from tools.web_search.models import SearchAttemptKind, WebSearchProviderRequest
from tools.web_search.source_policy import WebSourcePolicy


@dataclass(frozen=True)
class SearchAttemptPlan:
    """One progressive search attempt."""

    kind: SearchAttemptKind
    include_domains: tuple[str, ...]
    exclude_domains: tuple[str, ...]


class WebSearchQueryBuilder:
    """Build provider-neutral requests for each search attempt."""

    @staticmethod
    def plan_attempts(
        policy: WebSourcePolicy,
        *,
        allow_generic_fallback: bool,
        allow_exam_prep_fallback: bool,
        exam_prep_suitable: bool,
        official_only: bool,
        requires_fresh_evidence: bool = False,
    ) -> list[SearchAttemptPlan]:
        exclude = policy.global_blocked
        if policy.source_need == "practice_current_affairs" and not official_only:
            # Evidence-required current affairs must satisfy min_trusted_results, but a
            # merged trusted+reputed query lets high-volume news domains outrank
            # government ones so no official source ever surfaces. Ask for those alone
            # first; broad exam-prep digests keep the cheaper merged-first plan.
            attempts: list[SearchAttemptPlan] = []
            if requires_fresh_evidence and policy.trusted_domains:
                attempts.append(
                    SearchAttemptPlan(
                        kind="authoritative",
                        include_domains=policy.trusted_domains,
                        exclude_domains=exclude,
                    )
                )
            attempts.append(
                SearchAttemptPlan(
                    kind="authoritative_plus_reputed",
                    include_domains=tuple(
                        dict.fromkeys(
                            [*policy.trusted_domains, *policy.reputed_domains]
                        ).keys()
                    ),
                    exclude_domains=exclude,
                )
            )
        else:
            attempts = [
                SearchAttemptPlan(
                    kind="authoritative",
                    include_domains=policy.trusted_domains,
                    exclude_domains=exclude,
                ),
                SearchAttemptPlan(
                    kind="authoritative_plus_reputed",
                    include_domains=tuple(
                        dict.fromkeys(
                            [*policy.trusted_domains, *policy.reputed_domains]
                        ).keys()
                    ),
                    exclude_domains=exclude,
                ),
            ]
        if (
            allow_exam_prep_fallback
            and exam_prep_suitable
            and not official_only
            and policy.exam_prep_domains
        ):
            attempts.append(
                SearchAttemptPlan(
                    kind="exam_prep_fallback",
                    include_domains=policy.exam_prep_domains,
                    exclude_domains=exclude,
                )
            )
        if allow_generic_fallback:
            attempts.append(
                SearchAttemptPlan(
                    kind="generic_fallback",
                    include_domains=(),
                    exclude_domains=exclude,
                )
            )
        return attempts

    @staticmethod
    def build_provider_request(
        *,
        search_query: str,
        policy: WebSourcePolicy,
        attempt: SearchAttemptPlan,
        max_results: int,
        search_depth: str,
        timeout_seconds: float,
        requires_fresh_evidence: bool = False,
    ) -> WebSearchProviderRequest:
        # Freshness-required evidence is discarded without a publish date, and Tavily
        # only returns one for its "news" topic. Packs that prefer "finance"/"general"
        # would otherwise have every dated candidate rejected as undated.
        topic = "news" if requires_fresh_evidence else policy.topic
        return WebSearchProviderRequest(
            query=search_query,
            topic=topic,
            include_domains=list(attempt.include_domains),
            exclude_domains=list(attempt.exclude_domains),
            start_date=policy.start_date,
            end_date=policy.end_date,
            time_range=policy.time_range,
            max_results=max_results,
            search_depth=search_depth,
            include_raw_content=False,
            timeout_seconds=timeout_seconds,
            metadata={"source_pack": policy.source_pack_name, "attempt": attempt.kind},
        )
