from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GAConfig:
    # -------------------------
    # GA
    # -------------------------
    n_pop: int = 60
    num_parent: int = field(init=False)
    mutation_strength: float = 0.2
    seed: int = 123

    # -------------------------
    # Run control
    # -------------------------
    n_gen: int = 2
    stop_threshold: float = 0.995
    stop_mode: str = "fixed"         # "fixed" or "threshold"
    max_gen: Optional[int] = 150
    ic_mode: str = "absolute"        # "absolute" or "offset"

    # -------------------------
    # Scheduler / chunked HPC
    # -------------------------
    n_net_chunks: int = 50
    n_ic_chunks: int = 5
    n_ic_per_gen: int = 50
    ic_seed: Optional[int] = None
    keep_chunk_files: bool = False

    # -------------------------
    # Network
    # -------------------------
    input_size: int = 1
    hidden_size: int = 10
    output_size: int = 1

    # -------------------------
    # Time grid
    # -------------------------
    steps: int = 10000
    dense_steps: int = 20000
    total_time_ps: float = 20.0      # physical time in ps

    # -------------------------
    # Dynamics / solver
    # -------------------------
    integrator: str = "deterministic"   # "deterministic" or "baoab"
    method: str = "Radau"               # used only for deterministic solve_ivp
    rtol: float = 1e-3
    atol: float = 1e-6

    alpha: float = 8.75
    damping: float = 0.05               # legacy name; used as gamma if gamma is None
    gamma: Optional[float] = None       # Langevin friction coefficient

    # -------------------------
    # Thermostat / Langevin
    # -------------------------
    use_langevin: bool = False
    temperature_K: float = 300.0
    energy_conv_eV: float = 4.180159e-3   # 4.180159 meV = 0.004180159 eV
    kB_eV_per_K: float = 8.617333262e-5
    langevin_seed: Optional[int] = None
    thermostat_type: str = "langevin"

    # -------------------------
    # Field evaluation
    # -------------------------
    field_mode: str = "spline"        # "spline" or "nn"
    spline_k: int = 5
    spline_n: int = 400               # coarse grid points for spline/norm

    def __post_init__(self):
        self.num_parent = max(1, self.n_pop // 10)

        if self.stop_mode not in ("fixed", "threshold"):
            raise ValueError("stop_mode must be 'fixed' or 'threshold'")

        if self.ic_mode not in ("absolute", "offset"):
            raise ValueError("ic_mode must be 'absolute' or 'offset'")

        if self.field_mode not in ("spline", "nn"):
            raise ValueError("field_mode must be 'spline' or 'nn'")

        if self.integrator not in ("deterministic", "baoab"):
            raise ValueError("integrator must be 'deterministic' or 'baoab'")

        if self.thermostat_type != "langevin":
            raise ValueError("thermostat_type currently supports only 'langevin'")

        if self.gamma is None:
            self.gamma = self.damping

        if self.gamma < 0:
            raise ValueError("gamma must be >= 0")

        if self.temperature_K < 0:
            raise ValueError("temperature_K must be >= 0")

        if self.energy_conv_eV <= 0:
            raise ValueError("energy_conv_eV must be > 0")

        if self.steps < 2 or self.dense_steps < 2:
            raise ValueError("steps and dense_steps must be at least 2")

        if self.n_net_chunks <= 0:
            raise ValueError("n_net_chunks must be >= 1")

        if self.n_ic_chunks <= 0:
            raise ValueError("n_ic_chunks must be >= 1")

        if self.n_ic_per_gen <= 0:
            raise ValueError("n_ic_per_gen must be >= 1")

        if self.n_pop % self.n_net_chunks != 0:
            raise ValueError("n_pop must be divisible by n_net_chunks")

        if self.n_ic_per_gen % self.n_ic_chunks != 0:
            raise ValueError("n_ic_per_gen must be divisible by n_ic_chunks")

        # keep these consistent
        if self.integrator == "baoab":
            self.use_langevin = True

    @property
    def kBT_eV(self) -> float:
        return self.kB_eV_per_K * self.temperature_K

    @property
    def kBT_internal(self) -> float:
        return self.kBT_eV / self.energy_conv_eV

    @property
    def thermostat_seed(self) -> int:
        return self.seed if self.langevin_seed is None else self.langevin_seed

    @property
    def effective_ic_seed(self) -> int:
        return self.seed if self.ic_seed is None else self.ic_seed
