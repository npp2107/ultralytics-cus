# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

from .predict import DetectionPredictor
from .train import DetectionTrainer
from .val import DetectionValidator
from .gta_kd_train import GTA_KD_Trainer
from .kd_train import KD_Trainer
from .kd_val import KD_DetectionValidator
__all__ = "DetectionPredictor", "DetectionTrainer", "DetectionValidator", "KD_Trainer", "GTA_KD_Trainer", "KD_DetectionValidator"
