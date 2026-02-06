from typing import Literal

from pydantic import BaseModel, Field, model_validator


class SimConfig(BaseModel):
    """
    Configuration container for Cancer Growth Simulation.
    Validates stability criteria upon initialization.
    """

    DIM: Literal[2, 3] = Field(2, description="Spatial dimensionality (2 or 3)")
    L: float = Field(10.0, gt=0, description="Domain size in cm")
    N: int = Field(100, gt=10, description="Grid points per dimension")
    T_max: float = Field(20.0, gt=0, description="Total simulation time in days")

    dt: float = Field(0.01, gt=0, description="Time step in days")

    D_coeff: float = Field(0.05, ge=0, description="Diffusion coefficient (cm^2/day)")
    rho: float = Field(0.1, ge=0, description="Proliferation rate (1/day)")
    K: float = Field(1.0, gt=0, description="Carrying capacity")
    beta: float = Field(0.5, ge=0, description="Radiosensitivity")

    rad_start: float = 10.0
    rad_end: float = 15.0

    verbose: bool = Field(False, description="Set to true to enable verbose logging.")

    @property
    def dx(self) -> float:
        return self.L / self.N

    @model_validator(mode="after")
    def check_stability(self) -> "SimConfig":
        """
        Validates the Von Neumann stability condition for the diffusion term.
        """

        limit = (self.dx**2) / (2 * self.D * self.dim)

        if self.dt > limit:
            raise ValueError(
                f"\n UNSTABLE SIMULATION DETECTED!\n"
                f"Given dt ({self.dt}) exceeds the stability limit ({limit:.5f}).\n"
                f"Constraint: dt <= dx^2 / (2 * D * {self.dim})\n"
                f"Increase N or decrease dt."
            )

        if self.verbose:
            print(f"Configuration Valid. Stability limit: {limit:.5f} > dt: {self.dt}")

        return self

    def print_message(self, msg: str):
        """
        Logs the message if verbose variable is set to true.
        """

        if self.verbose:
            print(msg)


if __name__ == "__main__":
    s = SimConfig(DIM=2)
