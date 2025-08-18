# models.py
# Core data structures for ECA deterministic & probabilistic analyses
# Author: Ikechukwu Onyegiri + Copilot
# Python 3.10+

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Callable
import numpy as np
import pandas as pd


# ----------------------------
# Enums and simple identifiers
# ----------------------------
class FlawType(Enum):
    SURFACE_OD = "surface_od"
    SURFACE_ID = "surface_id"
    EMBEDDED_NEAR_OD = "embedded_near_od"
    EMBEDDED_MIDWALL = "embedded_midwall"
    EMBEDDED_NEAR_ID = "embedded_near_id"


class Phase(Enum):
    REELING = "reeling"          # ductile tearing
    INSTALLATION = "installation" # installation fatigue crack growth
    OPERATION = "operation"       # service fatigue crack growth


class FADOption(Enum):
    OPTION_1 = "option_1"  # canonical BS 7910 Option 1 (material independent)
    OPTION_2 = "option_2"  # material-specific from stress–strain


class Distribution(Enum):
    DETERM = "deterministic"
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    UNIFORM = "uniform"
    TRIANGULAR = "triangular"


# ----------------------------
# Material & curve definitions
# ----------------------------
@dataclass
class StressStrainPoint:
    strain: float  # true or engineering (must be consistent across dataset)
    stress: float  # MPa


@dataclass
class StressStrainCurve:
    """Material stress–strain curve for deriving Option 2 FAD and flow stress."""
    points: List[StressStrainPoint] = field(default_factory=list)
    is_true: bool = False  # if False, treat as engineering SS; conversion may be applied
    def as_arrays(self) -> Tuple[np.ndarray, np.ndarray]:
        eps = np.array([p.strain for p in self.points], dtype=float)
        sig = np.array([p.stress for p in self.points], dtype=float)
        return eps, sig

    def yield_strength(self) -> float:
        """Return 0.2% proof strength in MPa (simple approximate)."""
        eps, sig = self.as_arrays()
        # crude: pick first stress where strain >= 0.002 (eng) or 0.002 true
        idx = np.argmax(eps >= 0.002)
        return float(sig[idx])

    def ultimate_strength(self) -> float:
        _, sig = self.as_arrays()
        return float(sig.max())

    def flow_strength(self, p: float = 0.5) -> float:
        """Return weighted flow strength (e.g., 0.5*(YS+UTS))."""
        return p * self.yield_strength() + (1.0 - p) * self.ultimate_strength()


@dataclass
class JRCurvePoint:
    da: float   # crack extension Δa [mm]
    J: float    # J [kJ/m^2] or [N/mm]? We'll use N/mm (kJ/m^2 == N/mm)
    # Note: 1 kJ/m^2 = 1 N/mm


@dataclass
class JRCurve:
    """Project lower-bound JR-curve (e.g., SENT for reeling)."""
    points: List[JRCurvePoint] = field(default_factory=list)
    origin_shift_da: float = 0.0  # allow origin shift if needed

    def J_at_da(self, da: float) -> float:
        if not self.points:
            return np.nan
        xs = np.array([p.da for p in self.points], dtype=float)
        ys = np.array([p.J for p in self.points], dtype=float)
        return float(np.interp(da, xs, ys, left=ys[0], right=ys[-1]))


@dataclass
class FCGRParams:
    """Paris law parameters for da/dN = C*(ΔK)^m with optional ΔK_th and cap."""
    C: float
    m: float
    deltaK_th: float = 0.0   # threshold [MPa*sqrt(m)]
    deltaK_cap: float = np.inf
    env_factor: float = 1.0  # multiplicative factor for environment/KDF
# ----------------------------
# Geometry & load definitions
# ----------------------------

@dataclass
class PipeGeometry:
    t: float             # wall thickness [mm]
    OD: float            # outer diameter [mm]

    @property
    def ID(self) -> float:
        return self.OD - 2.0 * self.t

    @property
    def radius_mean(self) -> float:
        return 0.5 * (self.OD + self.ID) / 2.0


@dataclass
class FlawGeometry:
    """Semi-elliptical surface or embedded flaw."""
    a: float             # depth [mm]
    c: float             # half-length along weld [mm]
    ligament: Optional[float] = None  # for embedded flaws: distance from tip to nearest surface [mm]
    flaw_type: FlawType = FlawType.SURFACE_OD
    def is_surface(self) -> bool:
        return self.flaw_type in (FlawType.SURFACE_OD, FlawType.SURFACE_ID)

    def copy(self) -> "FlawGeometry":
        return FlawGeometry(a=self.a, c=self.c, ligament=self.ligament, flaw_type=self.flaw_type)


@dataclass
class Loading:
    """Membrane & bending stresses per phase; for reeling, use equivalent strain events."""
    sigma_m: float = 0.0   # membrane [MPa]
    sigma_b: float = 0.0   # bending [MPa]
    pressure: float = 0.0  # internal pressure [MPa] for biaxiality effects if needed
    def combined(self, k_b: float = 1.0) -> float:
        """Return linearized combined stress, simplistic."""
        return self.sigma_m + k_b * self.sigma_b


@dataclass
class ReelingStrainEvent:
    """Describes one tensile reeling event's driving input (e.g., reel-on or aligner)."""
    name: str
    axial_strain: float           # peak tensile engineering strain at weld [%] as decimal, e.g., 0.015 for 1.5%
    neutral_axis_shift: float = 0.0  # placeholder for eccentricity effects
    cycles: int = 1


# ----------------------------
# Deterministic input bundle
# ----------------------------
@dataclass
class DeterministicInputs:
    pipe: PipeGeometry
    flaw0: FlawGeometry
    mat_curve: StressStrainCurve
    fad_option: FADOption = FADOption.OPTION_2
    jr_curve_reeling: Optional[JRCurve] = None
    fcgr_install: Optional[FCGRParams] = None
    fcgr_operation: Optional[FCGRParams] = None
    reeling_events: List[ReelingStrainEvent] = field(default_factory=list)
    install_histogram: Optional[pd.DataFrame] = None  # columns: ['delta_sigma','cycles'] or ['deltaK','cycles']
    operation_histogram: Optional[pd.DataFrame] = None
    install_loading: Optional[Loading] = None
    operation_loading: Optional[Loading] = None
    poisson: float = 0.3
    youngs_modulus: float = 207000.0  # MPa
    tearing_limit_fraction_t: float = 0.10  # default 10% WT (configurable)
    rechar_rule_ligament_over_a: float = 0.5  # default 0.5 a for embedded->surface

    # NEW (optional toughness inputs for EOL check):
    Kmat: Optional[float] = None   # MPa√m
    Jc: Optional[float] = None     # N/mm (kJ/m^2)
    
# Optional helper (handy in engines/UI)
    def equivalent_Kmat(self) -> Optional[float]:
        """Return Kmat (MPa√m) using Jc if Kmat is not set; else return Kmat."""
        import numpy as np
        if self.Kmat is not None:
            return float(self.Kmat)
        if self.Jc is not None:
            E_prime = self.youngs_modulus / max(1.0 - self.poisson**2, 1e-9)
            return float(np.sqrt(self.Jc * E_prime))
        return None

# ----------------------------
# Probabilistic inputs
# ----------------------------

@dataclass
class RandomVar:
    name: str
    dist: Distribution
    params: Dict[str, float]  # e.g., {"mean":..., "std":...} or {"low":..., "high":...}
    transform: Optional[Callable[[np.ndarray], np.ndarray]] = None  # optional mapping

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        if self.dist == Distribution.DETERM:
            val = self.params.get("value", 0.0)
            return np.full(n, val, dtype=float)
        if self.dist == Distribution.NORMAL:
            x = rng.normal(self.params["mean"], self.params["std"], size=n)
        elif self.dist == Distribution.LOGNORMAL:
            # params: mean, std in log-space or arithmetic? we allow mu, sigma in log
            x = rng.lognormal(self.params["mu"], self.params["sigma"], size=n)
        elif self.dist == Distribution.UNIFORM:
            x = rng.uniform(self.params["low"], self.params["high"], size=n)
        elif self.dist == Distribution.TRIANGULAR:
            x = rng.triangular(self.params["left"], self.params["mode"], self.params["right"], size=n)
        else:
            raise ValueError(f"Unsupported distribution {self.dist}")
        if self.transform:
            x = self.transform(x)
        return x
@dataclass
class ProbabilisticInputs:
    """PFM study definition."""
    n_samples: int
    seed: int = 42
    # R.V.s for key inputs (flaw, material, geometry, loads, toughness, FCGR):
    rv_flaw_a: RandomVar = field(default_factory=lambda: RandomVar("a", Distribution.DETERM, {"value": 1.0}))
    rv_flaw_c: RandomVar = field(default_factory=lambda: RandomVar("c", Distribution.DETERM, {"value": 5.0}))
    rv_ligament: Optional[RandomVar] = None
    rv_Y: RandomVar = field(default_factory=lambda: RandomVar("YS", Distribution.DETERM, {"value": 450.0}))
    rv_UTS: RandomVar = field(default_factory=lambda: RandomVar("UTS", Distribution.DETERM, {"value": 550.0}))
    rv_t: RandomVar = field(default_factory=lambda: RandomVar("t", Distribution.DETERM, {"value": 20.0}))
    rv_OD: RandomVar = field(default_factory=lambda: RandomVar("OD", Distribution.DETERM, {"value": 273.0}))
    rv_Kmat: Optional[RandomVar] = None  # or JR params via J0, slope, etc.
    rv_sigma_install: RandomVar = field(default_factory=lambda: RandomVar("sigma_install", Distribution.DETERM, {"value": 0.0}))
    rv_sigma_oper: RandomVar = field(default_factory=lambda: RandomVar("sigma_oper", Distribution.DETERM, {"value": 0.0}))
    rv_fcgr_C: Optional[RandomVar] = None
    rv_fcgr_m: Optional[RandomVar] = None
    # deterministic context needed by the engine
    base: DeterministicInputs = field(default_factory=DeterministicInputs)

