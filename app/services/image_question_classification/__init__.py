"""Image-question classification application boundary."""

from services.image_question_classification.classifier import ImageQuestionClassifier
from services.image_question_classification.provider import ImageQuestionClassificationProvider

__all__ = ["ImageQuestionClassificationProvider", "ImageQuestionClassifier"]
