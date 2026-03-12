import numpy as np
import pandas as pd

from core.data_model import TransportDataset
from core.unit_conversion import convert_resistance_to_resistivity

# -------------------------------------------------
# Create dummy dataframe
# -------------------------------------------------

df = pd.DataFrame({
    "B_field": [-1, 0, 1],
    "rho_xx": [1e-6, 1.1e-6, 1.2e-6],
    "rho_xy": [-1e-7, 0, 1e-7]
})

dataset = TransportDataset.from_dataframe(df, temperature=2.0)

print("Dataset loaded")
print(dataset)

# -------------------------------------------------
# Test conversion
# -------------------------------------------------

R = np.array([10, 11, 12])  # Ohm

rho = convert_resistance_to_resistivity(
    R,
    length=0.001,
    width=0.0005,
    thickness=1e-6
)

print("Resistivity:", rho)