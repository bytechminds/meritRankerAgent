from features.practice_generation.option_distribution import (
    correct_option_index,
    reorder_options,
    target_correct_positions,
    validate_answer_position_distribution,
)


def test_five_question_option_rebalance_is_deterministic_and_balanced() -> None:
    test_id = "practice-test"
    question_ids = [f"q-{index}" for index in range(1, 6)]
    correct_answers = ["20%", "100000", "35% increase", "5.5% loss", "1080"]
    questions = []
    for question_id, correct_answer, target in zip(
        question_ids,
        correct_answers,
        target_correct_positions(test_id, question_ids),
        strict=True,
    ):
        options = [correct_answer, "distractor-one", "distractor-two", "distractor-three"]
        reordered = reorder_options(
            test_id=test_id,
            question_id=question_id,
            options=options,
            correct_answer=correct_answer,
            target_position=target,
        )
        assert reordered is not None
        assert correct_option_index(reordered, correct_answer) == target
        assert reorder_options(
            test_id=test_id,
            question_id=question_id,
            options=options,
            correct_answer=correct_answer,
            target_position=target,
        ) == reordered
        questions.append({"options": reordered, "correctAnswer": correct_answer})

    validation = validate_answer_position_distribution(questions)

    assert validation.valid is True
    assert sorted(
        validation.correct_positions.count(position) for position in range(4)
    ) == [1, 1, 1, 2]


def test_answer_position_validation_rejects_a_five_question_first_option_pattern() -> None:
    validation = validate_answer_position_distribution(
        [
            {"options": ["correct", "b", "c", "d"], "correctAnswer": "correct"}
            for _ in range(5)
        ]
    )

    assert (validation.valid, validation.reason_code) == (
        False,
        "ANSWER_POSITION_DISTRIBUTION_INVALID",
    )
