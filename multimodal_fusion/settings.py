from pathlib import Path
ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
DATA = ROOT / "data"
FEATURES = ["motor_pd_probability", "voice_pd_probability", "handwriting_pd_probability"]
LABELS = ["PD", "ET", "CA", "AP", "DLB"]
NAMES = {"PD": "Parkinson's Disease", "ET": "Essential Tremor", "CA": "Cerebellar Ataxia", "AP": "Atypical Parkinsonism", "DLB": "Dementia with Lewy Bodies"}
SEED = 401
VERSION = "3.0.0-pp1-moderate"
ACTIVE_DATASET = "synthetic_moderate_participants.csv"
SCOPE = "Synthetic PP1 demonstration: moderate-overlap scenario (noise SD 0.50). Five-class outputs are simulator-based model estimates, not validated disease probabilities. The retained harder benchmark is reported separately. Healthy is not a supported class."
