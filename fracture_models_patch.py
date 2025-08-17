# --- at top of fracture_models.py, update imports ---
from models import (
    PipeGeometry, FlawGeometry, StressStrainCurve, JRCurve,
    FADOption, DeterministicInputs, # add FlawType to imports
    # NEW:
    FlawType
)

# --- in check_recharacterization(...) ---
def check_recharacterization(flaw: FlawGeometry, rule_ratio: float) -> FlawGeometry:
    """
    If ligament < rule_ratio * a, convert embedded flaw to surface flaw.
    """
    if flaw.ligament is not None and flaw.ligament < rule_ratio * flaw.a:
        new_flaw = flaw.copy()
        # FIX: reference FlawType enum, not FlawGeometry.FlawType
        new_flaw.flaw_type = FlawType.SURFACE_OD
        new_flaw.ligament = None
        return new_flaw
    return flaw
