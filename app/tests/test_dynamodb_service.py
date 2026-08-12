"""
app/tests/test_dynamodb_service.py
------------------------------------
Unit tests for app/services/dynamodb_service.py.

All tests mock the DynamoDB client — no real AWS calls, no credentials required.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

import config as cfg_module
from services.dynamodb_service import (
    DynamoDbServiceError,
    _from_dynamodb_item,
    _from_dynamodb_value,
    _key_to_dynamodb,
    _to_dynamodb_key_value,
    batch_get_items,
    get_item,
    query_by_index,
    query_by_partition_key,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reset_settings():
    cfg_module._settings = None


def _make_client_error(code: str = "InternalServerError") -> ClientError:
    return ClientError(
        error_response={"Error": {"Code": code, "Message": "test error"}},
        operation_name="TestOp",
    )


def _dynamo_str(value: str) -> dict:
    return {"S": value}


def _dynamo_num(value: str) -> dict:
    return {"N": value}


def _mock_client() -> MagicMock:
    return MagicMock()


# ---------------------------------------------------------------------------
# AttributeValue conversion — _from_dynamodb_value
# ---------------------------------------------------------------------------


class TestFromDynamodbValue:
    def test_string_type(self):
        assert _from_dynamodb_value({"S": "hello"}) == "hello"

    def test_number_integer(self):
        result = _from_dynamodb_value({"N": "42"})
        assert result == 42
        assert isinstance(result, int)

    def test_number_float(self):
        result = _from_dynamodb_value({"N": "3.14"})
        assert result == pytest.approx(3.14)
        assert isinstance(result, float)

    def test_number_scientific_notation(self):
        result = _from_dynamodb_value({"N": "1e5"})
        assert result == pytest.approx(100000.0)

    def test_boolean_true(self):
        assert _from_dynamodb_value({"BOOL": True}) is True

    def test_boolean_false(self):
        assert _from_dynamodb_value({"BOOL": False}) is False

    def test_null_type(self):
        assert _from_dynamodb_value({"NULL": True}) is None

    def test_list_type(self):
        result = _from_dynamodb_value({"L": [{"S": "a"}, {"N": "1"}]})
        assert result == ["a", 1]

    def test_map_type(self):
        result = _from_dynamodb_value({"M": {"key": {"S": "value"}, "num": {"N": "5"}}})
        assert result == {"key": "value", "num": 5}

    def test_string_set(self):
        result = _from_dynamodb_value({"SS": ["a", "b", "c"]})
        assert set(result) == {"a", "b", "c"}

    def test_number_set(self):
        result = _from_dynamodb_value({"NS": ["1", "2", "3"]})
        assert set(result) == {1, 2, 3}

    def test_binary_type_returns_bytes(self):
        data = b"raw bytes"
        result = _from_dynamodb_value({"B": data})
        assert result == data

    def test_unknown_type_returns_none(self):
        assert _from_dynamodb_value({"UNKNOWN": "value"}) is None

    def test_nested_map_in_list(self):
        result = _from_dynamodb_value({"L": [{"M": {"x": {"S": "y"}}}]})
        assert result == [{"x": "y"}]


class TestFromDynamodbItem:
    def test_converts_full_item(self):
        item = {
            "question_id": {"S": "q-001"},
            "difficulty": {"N": "3"},
            "active": {"BOOL": True},
            "tags": {"SS": ["math", "algebra"]},
        }
        result = _from_dynamodb_item(item)
        assert result["question_id"] == "q-001"
        assert result["difficulty"] == 3
        assert result["active"] is True
        assert set(result["tags"]) == {"math", "algebra"}


# ---------------------------------------------------------------------------
# _to_dynamodb_key_value
# ---------------------------------------------------------------------------


class TestToDynamodbKeyValue:
    def test_string_becomes_S(self):
        assert _to_dynamodb_key_value("hello") == {"S": "hello"}

    def test_integer_becomes_N(self):
        assert _to_dynamodb_key_value(42) == {"N": "42"}

    def test_float_becomes_N(self):
        assert _to_dynamodb_key_value(3.14) == {"N": "3.14"}

    def test_boolean_raises_type_error(self):
        with pytest.raises(TypeError):
            _to_dynamodb_key_value(True)

    def test_none_raises_type_error(self):
        with pytest.raises(TypeError):
            _to_dynamodb_key_value(None)

    def test_list_raises_type_error(self):
        with pytest.raises(TypeError):
            _to_dynamodb_key_value(["a", "b"])


class TestKeyToDynamodb:
    def test_single_string_key(self):
        result = _key_to_dynamodb({"question_id": "q-001"})
        assert result == {"question_id": {"S": "q-001"}}

    def test_composite_key(self):
        result = _key_to_dynamodb({"pk": "partition", "sk": "sort"})
        assert result == {"pk": {"S": "partition"}, "sk": {"S": "sort"}}


# ---------------------------------------------------------------------------
# get_item
# ---------------------------------------------------------------------------


class TestGetItem:
    def _setup(self, monkeypatch):
        monkeypatch.setenv("DYNAMODB_REGION", "")
        _reset_settings()

    def test_returns_parsed_item_when_found(self, monkeypatch):
        self._setup(monkeypatch)
        mock_client = _mock_client()
        mock_client.get_item.return_value = {
            "Item": {
                "question_id": {"S": "q-001"},
                "text": {"S": "What is algebra?"},
                "difficulty": {"N": "2"},
            }
        }

        with patch("services.dynamodb_service.get_dynamodb_client", return_value=mock_client):
            result = get_item("questions-table", {"question_id": "q-001"})

        assert result == {"question_id": "q-001", "text": "What is algebra?", "difficulty": 2}

    def test_returns_none_when_item_not_found(self, monkeypatch):
        self._setup(monkeypatch)
        mock_client = _mock_client()
        mock_client.get_item.return_value = {}  # No "Item" key

        with patch("services.dynamodb_service.get_dynamodb_client", return_value=mock_client):
            result = get_item("questions-table", {"question_id": "missing"})

        assert result is None

    def test_passes_correct_request_to_client(self, monkeypatch):
        self._setup(monkeypatch)
        mock_client = _mock_client()
        mock_client.get_item.return_value = {}

        with patch("services.dynamodb_service.get_dynamodb_client", return_value=mock_client):
            get_item("my-table", {"question_id": "q-99"})

        call_kwargs = mock_client.get_item.call_args[1]
        assert call_kwargs["TableName"] == "my-table"
        assert call_kwargs["Key"] == {"question_id": {"S": "q-99"}}

    def test_client_error_raises_dynamodb_service_error(self, monkeypatch):
        self._setup(monkeypatch)
        mock_client = _mock_client()
        mock_client.get_item.side_effect = _make_client_error("ResourceNotFoundException")

        with patch("services.dynamodb_service.get_dynamodb_client", return_value=mock_client):
            with pytest.raises(DynamoDbServiceError, match="ResourceNotFoundException"):
                get_item("missing-table", {"question_id": "q-1"})


# ---------------------------------------------------------------------------
# batch_get_items
# ---------------------------------------------------------------------------


class TestBatchGetItems:
    def _setup(self, monkeypatch):
        monkeypatch.setenv("DYNAMODB_REGION", "")
        _reset_settings()

    def test_returns_empty_list_for_empty_input(self, monkeypatch):
        self._setup(monkeypatch)
        with patch("services.dynamodb_service.get_dynamodb_client") as mock_factory:
            result = batch_get_items("table", [])

        mock_factory.assert_not_called()
        assert result == []

    def test_returns_parsed_items(self, monkeypatch):
        self._setup(monkeypatch)
        mock_client = _mock_client()
        mock_client.batch_get_item.return_value = {
            "Responses": {
                "patterns-table": [
                    {"pattern_id": {"S": "p-1"}, "title": {"S": "Linear equations"}},
                    {"pattern_id": {"S": "p-2"}, "title": {"S": "Quadratics"}},
                ]
            }
        }

        with patch("services.dynamodb_service.get_dynamodb_client", return_value=mock_client):
            results = batch_get_items(
                "patterns-table",
                [{"pattern_id": "p-1"}, {"pattern_id": "p-2"}],
            )

        assert len(results) == 2
        assert results[0]["pattern_id"] == "p-1"
        assert results[1]["title"] == "Quadratics"

    def test_passes_correct_keys_to_client(self, monkeypatch):
        self._setup(monkeypatch)
        mock_client = _mock_client()
        mock_client.batch_get_item.return_value = {"Responses": {"t": []}}

        with patch("services.dynamodb_service.get_dynamodb_client", return_value=mock_client):
            batch_get_items("t", [{"question_id": "q-1"}, {"question_id": "q-2"}])

        call_kwargs = mock_client.batch_get_item.call_args[1]
        keys = call_kwargs["RequestItems"]["t"]["Keys"]
        assert keys == [{"question_id": {"S": "q-1"}}, {"question_id": {"S": "q-2"}}]

    def test_client_error_raises_service_error(self, monkeypatch):
        self._setup(monkeypatch)
        mock_client = _mock_client()
        mock_client.batch_get_item.side_effect = _make_client_error("ThrottlingException")

        with patch("services.dynamodb_service.get_dynamodb_client", return_value=mock_client):
            with pytest.raises(DynamoDbServiceError, match="ThrottlingException"):
                batch_get_items("t", [{"pattern_id": "p-1"}])

    def test_batch_size_capped_at_max(self, monkeypatch):
        self._setup(monkeypatch)
        mock_client = _mock_client()
        mock_client.batch_get_item.return_value = {"Responses": {"t": []}}

        # Provide 150 keys — should be capped at 100 (_MAX_BATCH_SIZE)
        keys = [{"question_id": str(i)} for i in range(150)]
        with patch("services.dynamodb_service.get_dynamodb_client", return_value=mock_client):
            batch_get_items("t", keys)

        call_kwargs = mock_client.batch_get_item.call_args[1]
        sent_keys = call_kwargs["RequestItems"]["t"]["Keys"]
        assert len(sent_keys) == 100

    def test_missing_table_in_response_returns_empty(self, monkeypatch):
        self._setup(monkeypatch)
        mock_client = _mock_client()
        mock_client.batch_get_item.return_value = {"Responses": {}}  # table absent

        with patch("services.dynamodb_service.get_dynamodb_client", return_value=mock_client):
            result = batch_get_items("t", [{"question_id": "q-1"}])

        assert result == []

    def test_default_does_not_retry_unprocessed_keys(self, monkeypatch):
        self._setup(monkeypatch)
        mock_client = _mock_client()
        mock_client.batch_get_item.return_value = {
            "Responses": {"t": [{"question_id": {"S": "q-1"}}]},
            "UnprocessedKeys": {
                "t": {"Keys": [{"question_id": {"S": "q-2"}}]}
            },
        }

        with patch("services.dynamodb_service.get_dynamodb_client", return_value=mock_client):
            result = batch_get_items("t", [{"question_id": "q-1"}, {"question_id": "q-2"}])

        assert result == [{"question_id": "q-1"}]
        assert mock_client.batch_get_item.call_count == 1

    def test_opt_in_retries_only_unprocessed_keys(self, monkeypatch):
        self._setup(monkeypatch)
        mock_client = _mock_client()
        first_key = {"question_id": {"S": "q-1"}}
        second_key = {"question_id": {"S": "q-2"}}
        mock_client.batch_get_item.side_effect = [
            {
                "Responses": {"t": [first_key]},
                "UnprocessedKeys": {"t": {"Keys": [second_key]}},
            },
            {"Responses": {"t": [second_key]}},
        ]

        with (
            patch("services.dynamodb_service.get_dynamodb_client", return_value=mock_client),
            patch("services.dynamodb_service.time.sleep") as mock_sleep,
        ):
            result = batch_get_items(
                "t",
                [{"question_id": "q-1"}, {"question_id": "q-2"}],
                max_unprocessed_retries=2,
            )

        assert result == [{"question_id": "q-1"}, {"question_id": "q-2"}]
        assert mock_client.batch_get_item.call_count == 2
        assert mock_client.batch_get_item.call_args_list[1].kwargs["RequestItems"] == {
            "t": {"Keys": [second_key]}
        }
        mock_sleep.assert_called_once_with(0.05)

    def test_unprocessed_key_retries_are_bounded(self, monkeypatch):
        self._setup(monkeypatch)
        mock_client = _mock_client()
        key = {"question_id": {"S": "q-1"}}
        response = {"Responses": {"t": []}, "UnprocessedKeys": {"t": {"Keys": [key]}}}
        mock_client.batch_get_item.side_effect = [response, response, response]

        with (
            patch("services.dynamodb_service.get_dynamodb_client", return_value=mock_client),
            patch("services.dynamodb_service.time.sleep"),
        ):
            result = batch_get_items(
                "t", [{"question_id": "q-1"}], max_unprocessed_retries=2
            )

        assert result == []
        assert mock_client.batch_get_item.call_count == 3

    @pytest.mark.parametrize("retry_count", [-1, 3, True, "1"])
    def test_rejects_invalid_unprocessed_key_retry_limit(self, monkeypatch, retry_count):
        self._setup(monkeypatch)

        with pytest.raises(ValueError, match="max_unprocessed_retries"):
            batch_get_items("t", [{"question_id": "q-1"}], max_unprocessed_retries=retry_count)


# ---------------------------------------------------------------------------
# query_by_partition_key
# ---------------------------------------------------------------------------


class TestQueryByPartitionKey:
    def _setup(self, monkeypatch):
        monkeypatch.setenv("DYNAMODB_REGION", "")
        _reset_settings()

    def test_returns_parsed_items(self, monkeypatch):
        self._setup(monkeypatch)
        mock_client = _mock_client()
        mock_client.query.return_value = {
            "Items": [
                {"question_id": {"S": "q-1"}, "text": {"S": "First question"}},
            ],
            "Count": 1,
        }

        with patch("services.dynamodb_service.get_dynamodb_client", return_value=mock_client):
            results = query_by_partition_key("t", "question_id", "q-1")

        assert len(results) == 1
        assert results[0]["question_id"] == "q-1"

    def test_passes_correct_kwargs_to_client(self, monkeypatch):
        self._setup(monkeypatch)
        mock_client = _mock_client()
        mock_client.query.return_value = {"Items": [], "Count": 0}

        with patch("services.dynamodb_service.get_dynamodb_client", return_value=mock_client):
            query_by_partition_key("t", "question_id", "q-1", limit=5)

        call_kwargs = mock_client.query.call_args[1]
        assert call_kwargs["TableName"] == "t"
        assert call_kwargs["ExpressionAttributeNames"] == {"#pk": "question_id"}
        assert call_kwargs["ExpressionAttributeValues"] == {":pk_val": {"S": "q-1"}}
        assert call_kwargs["Limit"] == 5
        assert "IndexName" not in call_kwargs

    def test_passes_index_name_when_provided(self, monkeypatch):
        self._setup(monkeypatch)
        mock_client = _mock_client()
        mock_client.query.return_value = {"Items": [], "Count": 0}

        with patch("services.dynamodb_service.get_dynamodb_client", return_value=mock_client):
            query_by_partition_key("t", "topic", "algebra", index_name="topic-index")

        call_kwargs = mock_client.query.call_args[1]
        assert call_kwargs["IndexName"] == "topic-index"

    def test_limit_capped_at_max(self, monkeypatch):
        self._setup(monkeypatch)
        mock_client = _mock_client()
        mock_client.query.return_value = {"Items": [], "Count": 0}

        with patch("services.dynamodb_service.get_dynamodb_client", return_value=mock_client):
            query_by_partition_key("t", "k", "v", limit=999)

        call_kwargs = mock_client.query.call_args[1]
        assert call_kwargs["Limit"] == 50  # _MAX_QUERY_LIMIT

    def test_client_error_raises_service_error(self, monkeypatch):
        self._setup(monkeypatch)
        mock_client = _mock_client()
        mock_client.query.side_effect = _make_client_error("AccessDeniedException")

        with patch("services.dynamodb_service.get_dynamodb_client", return_value=mock_client):
            with pytest.raises(DynamoDbServiceError, match="AccessDeniedException"):
                query_by_partition_key("t", "k", "v")

    def test_empty_items_response_returns_empty_list(self, monkeypatch):
        self._setup(monkeypatch)
        mock_client = _mock_client()
        mock_client.query.return_value = {"Items": [], "Count": 0}

        with patch("services.dynamodb_service.get_dynamodb_client", return_value=mock_client):
            result = query_by_partition_key("t", "k", "v")

        assert result == []


# ---------------------------------------------------------------------------
# query_by_index
# ---------------------------------------------------------------------------


class TestQueryByIndex:
    def _setup(self, monkeypatch):
        monkeypatch.setenv("DYNAMODB_REGION", "")
        _reset_settings()

    def test_delegates_to_query_with_index_name(self, monkeypatch):
        self._setup(monkeypatch)
        mock_client = _mock_client()
        mock_client.query.return_value = {
            "Items": [{"pattern_id": {"S": "p-1"}}],
            "Count": 1,
        }

        with patch("services.dynamodb_service.get_dynamodb_client", return_value=mock_client):
            results = query_by_index("t", "my-gsi", "topic", "algebra", limit=3)

        call_kwargs = mock_client.query.call_args[1]
        assert call_kwargs["IndexName"] == "my-gsi"
        assert call_kwargs["Limit"] == 3
        assert results[0]["pattern_id"] == "p-1"

    def test_projects_only_requested_attributes(self, monkeypatch):
        self._setup(monkeypatch)
        mock_client = _mock_client()
        mock_client.query.return_value = {"Items": [], "Count": 0}

        with patch("services.dynamodb_service.get_dynamodb_client", return_value=mock_client):
            query_by_index(
                "t",
                "my-gsi",
                "patternId",
                "pattern-1",
                projection_fields=("questionId", "questionText"),
            )

        call_kwargs = mock_client.query.call_args[1]
        assert call_kwargs["ProjectionExpression"] == "#projection_0, #projection_1"
        assert call_kwargs["ExpressionAttributeNames"] == {
            "#pk": "patternId",
            "#projection_0": "questionId",
            "#projection_1": "questionText",
        }
