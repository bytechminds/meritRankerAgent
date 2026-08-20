"""Inspect human-readable local agent request blocks."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

DEFAULT_LOG_PATH = Path(__file__).resolve().parents[1] / ".logs" / "agent-runtime.log"
DIVIDER = "=" * 80
_HEADER = re.compile(
    r"^REQUEST (?P<request_id>[^|]+) \| (?P<timestamp>[^|]+) \| "
    r"(?P<request_type>[^|]+) \| (?P<result>.+)$"
)
_IDENTITY = re.compile(
    r"^Conversation: (?P<conversation_id>\S+)\s+Turn: (?P<turn_id>\S+)\s+"
    r"Duration: (?P<duration>\S+)$"
)


@dataclass(frozen=True)
class RequestBlock:
    request_id: str
    trace_id: str
    conversation_id: str
    turn_id: str
    timestamp: str
    request_type: str
    result: str
    text: str


def load_blocks(path: Path) -> list[RequestBlock]:
    try:
        content = path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return []
    blocks: list[RequestBlock] = []
    for raw in content.split(DIVIDER):
        text = raw.strip()
        if not text:
            continue
        lines = text.splitlines()
        if len(lines) < 2:
            continue
        header = _HEADER.fullmatch(lines[0])
        identity = next(
            (_IDENTITY.fullmatch(line) for line in lines[1:6] if line.startswith("Conversation: ")),
            None,
        )
        if header is None or identity is None:
            continue
        trace_id = "-"
        trace_line = next((line for line in lines[1:7] if line.startswith("Trace: ")), None)
        if trace_line is not None:
            trace_id = trace_line.removeprefix("Trace: ").strip() or "-"
        blocks.append(
            RequestBlock(
                request_id=header["request_id"].strip(),
                trace_id=trace_id,
                conversation_id=identity["conversation_id"],
                turn_id=identity["turn_id"],
                timestamp=header["timestamp"].strip(),
                request_type=header["request_type"].strip(),
                result=header["result"].strip(),
                text=f"{text}\n{DIVIDER}",
            )
        )
    return blocks


def select_blocks(
    blocks: Iterable[RequestBlock],
    *,
    command: str,
    value: str | None,
    limit: int,
) -> list[RequestBlock]:
    selected = list(blocks)
    short_value = (value or "")[:8]
    if command == "latest":
        selected = selected[-1:]
    elif command == "request":
        selected = [block for block in selected if block.request_id == short_value]
    elif command == "conversation":
        selected = [block for block in selected if block.conversation_id == short_value]
    elif command == "failures":
        selected = [
            block
            for block in selected
            if block.result
            in {"failed", "failed-quality", "clarification", "client disconnect"}
        ]
    elif command == "persistence-failures":
        selected = [
            block
            for block in selected
            if any(
                marker in block.text
                for marker in (
                    "History write unavailable",
                    "Session update unavailable",
                    "Memory write unavailable",
                )
            )
        ]
    elif command == "follow-up-failures":
        selected = [
            block
            for block in selected
            if "Follow-up resolution failed" in block.text
            or "Recent context unavailable" in block.text
        ]
    return selected[-limit:]


@dataclass(frozen=True)
class OperationLine:
    """One incremental operation event or its terminal aggregate summary."""

    operation_id: str
    test_id: str
    execution_id: str
    request_id: str
    summary: bool
    text: str


_OPERATION_PREFIXES = ("[operation] ", "[operation-summary] ")


def _operation_field(text: str, key: str) -> str:
    match = re.search(rf"(?:^|\s){re.escape(key)}=(\S+)", text)
    return match.group(1) if match else ""


def load_operation_lines(path: Path) -> list[OperationLine]:
    """Read the operation timeline written by the long-running operation scope."""
    try:
        content = path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return []
    lines: list[OperationLine] = []
    for raw in content.splitlines():
        text = raw.strip()
        if not text.startswith(_OPERATION_PREFIXES):
            continue
        lines.append(
            OperationLine(
                operation_id=_operation_field(text, "op"),
                test_id=_operation_field(text, "test"),
                execution_id=_operation_field(text, "exec"),
                request_id=_operation_field(text, "req"),
                summary=text.startswith("[operation-summary] "),
                text=text,
            )
        )
    return lines


def select_operation_lines(
    lines: Iterable[OperationLine],
    *,
    operation_id: str | None = None,
    test_id: str | None = None,
    execution_id: str | None = None,
) -> list[OperationLine]:
    """Reconstruct exactly one operation timeline; identifiers combine with AND."""
    return [
        line
        for line in lines
        if (operation_id is None or line.operation_id == operation_id)
        and (test_id is None or line.test_id == test_id)
        and (execution_id is None or line.execution_id == execution_id)
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=DEFAULT_LOG_PATH)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument(
        "command",
        nargs="?",
        choices=(
            "latest",
            "request",
            "conversation",
            "failures",
            "persistence-failures",
            "follow-up-failures",
        ),
    )
    parser.add_argument("value", nargs="?")
    filters = parser.add_mutually_exclusive_group()
    filters.add_argument("--latest", action="store_true")
    filters.add_argument("--request-id")
    filters.add_argument("--conversation-id")
    filters.add_argument("--failures", action="store_true")
    filters.add_argument("--persistence-failures", action="store_true")
    filters.add_argument("--follow-up-failures", action="store_true")
    operations = parser.add_argument_group("operation timeline")
    operations.add_argument("--operation-id")
    operations.add_argument("--test-id")
    operations.add_argument("--execution-id")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.limit < 1:
        print("--limit must be greater than zero.", file=sys.stderr)
        return 2
    if args.operation_id or args.test_id or args.execution_id:
        timeline = select_operation_lines(
            load_operation_lines(args.path),
            operation_id=args.operation_id,
            test_id=args.test_id,
            execution_id=args.execution_id,
        )[-args.limit :]
        if args.as_json:
            print(json.dumps([asdict(line) for line in timeline], indent=2, sort_keys=True))
        else:
            for line in timeline:
                print(line.text)
        return 0
    command = args.command
    value = args.value
    if args.latest:
        command = "latest"
    elif args.request_id:
        command, value = "request", args.request_id
    elif args.conversation_id:
        command, value = "conversation", args.conversation_id
    elif args.failures:
        command = "failures"
    elif args.persistence_failures:
        command = "persistence-failures"
    elif args.follow_up_failures:
        command = "follow-up-failures"
    if command is None:
        print("Select one log filter.", file=sys.stderr)
        return 2
    if command in {"request", "conversation"} and not value:
        print(f"{command} requires an identifier.", file=sys.stderr)
        return 2

    selected = select_blocks(
        load_blocks(args.path),
        command=command,
        value=value,
        limit=args.limit,
    )
    if args.as_json:
        print(json.dumps([asdict(block) for block in selected], indent=2, sort_keys=True))
    else:
        for block in selected:
            print(block.text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
