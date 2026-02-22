from dataclasses import dataclass
from pathlib import Path

@dataclass
class Settings:
    IMG_SIZE: int = 224
    THRESHOLD: float = 0.5

    DATA_DIR: Path = Path("data/processed")
    WEIGHTS_DIR: Path = Path("weights")
    OUTPUTS_DIR: Path = Path("outputs")
    LOGS_DIR: Path = Path("logs")

    MODEL_A_PATH: Path = WEIGHTS_DIR / "model_a_effnet.pth"
    MODEL_B_PATH: Path = WEIGHTS_DIR / "model_b_resnet.pth"
    MODEL_C_PATH: Path = WEIGHTS_DIR / "model_c_cnnlstm.pth"

settings = Settings()
