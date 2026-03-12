import numpy as np

from utils.validation import (
    validate_dataset_structure,
    validate_no_nan,
    validate_monotonic_field
)

# Example experimental data
B = np.array([-1, 0, 1])
rho_xx = np.array([1e-6, 1.1e-6, 1.2e-6])
rho_xy = np.array([-1e-7, 0, 1e-7])

# Run validation checks
validate_dataset_structure(B, rho_xx, rho_xy)
validate_no_nan(B, rho_xx, rho_xy)
validate_monotonic_field(B)

print("Validation passed successfully")