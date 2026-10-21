"""otzyv-ml: классификатор тональности и спама для русских отзывов."""

from .model import Classifier, Prediction, build_pipeline, train_full
from .preprocess import normalize

__all__ = ["Classifier", "Prediction", "build_pipeline", "normalize", "train_full"]
__version__ = "1.0.0"
