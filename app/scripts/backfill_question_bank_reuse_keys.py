#!/usr/bin/env python3
"""One-time backfill of QuestionBank ``reuseBucketKey`` to the current slot identity.

Schema-v2 promotion used to key rows on the planner's per-run ``category_id``; the
current key uses the subject family (``matching.build_slot_reuse_bucket_key``). Rows
written before that change are invisible to reuse. This rewrites only
``reuseBucketKey``, derived deterministically from each row's own stored subject,
topic, question type and language through the production key builder.

A row is rewritten only when every check passes; anything else is left untouched:
- its current key is derivable from stored metadata;
- the stored key differs from it;
- ``reuseSortKey`` already carries the prefix for its stored difficulty;
- it passes the production reuse gate, including a non-empty solution.

Dry-run by default. ``--apply --table <name>`` must name the configured table. Each
write is conditional on the old key, so reruns and concurrent edits are safe.
Playable content, identity, trust and Pattern fields are never written.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from botocore.exceptions import ClientError  # noqa: E402

from features.practice_generation.matching import (  # noqa: E402
    build_reuse_bucket_key,
    build_reuse_difficulty_prefix,
    reusable_question_from_item,
)
from features.practice_generation.metadata_normalization import (  # noqa: E402
    normalize_language,
    normalize_subject,
    normalize_topic,
)

_DIFFICULTY = {"EASY": "basic", "MEDIUM": "intermediate", "HARD": "advanced"}


def _meta(item: dict[str, Any]) -> dict[str, Any]:
    value = item.get("meta") or {}
    return json.loads(value) if isinstance(value, str) else dict(value)


def plan_row(item: dict[str, Any]) -> tuple[str, str | None]:
    """Return ``(decision, current_key)`` for one plain QuestionBank row."""
    meta = _meta(item)
    subject = normalize_subject(meta.get("subject"))
    topic = normalize_topic(meta.get("topic"))
    language = normalize_language(meta.get("language"))
    question_type = str(meta.get("questionType") or "").casefold()
    if not (subject and topic and language and question_type):
        return "SKIP_UNDERIVABLE", None
    try:
        current = build_reuse_bucket_key(
            category=subject,
            topic=topic,
            question_type=question_type,
            language=language,
        )
    except ValueError:
        return "SKIP_UNDERIVABLE", None
    if item.get("reuseBucketKey") == current:
        return "ALREADY_CURRENT", current
    prefix = build_reuse_difficulty_prefix(_DIFFICULTY.get(str(item.get("difficulty")).upper()))
    if not prefix or not str(item.get("reuseSortKey") or "").startswith(prefix):
        return "SKIP_SORT_KEY_INCONSISTENT", current
    if reusable_question_from_item(item, requested_language=language) is None:
        return "SKIP_NOT_REUSABLE", current
    if not str(item.get("explanation") or "").strip():
        return "SKIP_NO_SOLUTION", current
    return "MIGRATE", current


def migrate_row(client: Any, *, table: str, item: dict[str, Any], current_key: str) -> bool:
    """Rewrite only ``reuseBucketKey``, and only if it still holds the planned old value."""
    try:
        client.update_item(
            TableName=table,
            Key={"qbId": {"S": str(item["qbId"])}},
            UpdateExpression="SET #key = :current",
            ConditionExpression="attribute_exists(qbId) AND #key = :old",
            ExpressionAttributeNames={"#key": "reuseBucketKey"},
            ExpressionAttributeValues={
                ":current": {"S": current_key},
                ":old": {"S": str(item["reuseBucketKey"])},
            },
        )
        return True
    except ClientError as exc:
        if exc.response.get("Error", {}).get("Code") == "ConditionalCheckFailedException":
            return False
        raise


def run(client: Any, *, table: str, apply: bool) -> Counter[str]:
    from features.practice_generation.repositories import _plain  # noqa: PLC0415

    counts: Counter[str] = Counter()
    kwargs: dict[str, Any] = {"TableName": table}
    while True:
        response = client.scan(**kwargs)
        for raw in response.get("Items", []):
            item = _plain(raw)
            decision, current_key = plan_row(item)
            counts[decision] += 1
            if decision == "MIGRATE" and apply and current_key is not None:
                migrated = migrate_row(client, table=table, item=item, current_key=current_key)
                counts["MIGRATED" if migrated else "CONDITION_FAILED"] += 1
        if "LastEvaluatedKey" not in response:
            return counts
        kwargs["ExclusiveStartKey"] = response["LastEvaluatedKey"]


def main() -> int:
    from features.practice_generation.config import get_practice_config  # noqa: PLC0415
    from services.aws_client_factory import get_dynamodb_client  # noqa: PLC0415

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--table", default=None, help="Required with --apply; must match config.")
    args = parser.parse_args()
    config = get_practice_config()
    if args.apply and args.table != config.question_bank_table:
        print("--apply requires --table equal to the configured QuestionBank table.")
        return 2
    counts = run(
        get_dynamodb_client(config.aws_region),
        table=config.question_bank_table,
        apply=args.apply,
    )
    print(json.dumps({"table": config.question_bank_table, "apply": args.apply, **counts}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
