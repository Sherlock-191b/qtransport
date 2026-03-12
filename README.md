[# Magnetotransport Data Analysis Toolkit (In Development)

A modular Python toolkit for **analysis of magnetotransport measurements in quantum materials**, including Hall transport, weak localization, and quantum oscillations.

This project aims to provide a **reproducible and extensible computational framework** for processing and analyzing experimental magnetotransport datasets commonly encountered in condensed matter physics research.

⚠️ **Status:** This project is currently **under active development**. APIs and features may change.

---

# Table of Contents

1. Project Motivation  
2. Features  
3. Physics Background  
4. Repository Structure  
5. Installation  
6. Usage  
7. Example Workflows  
8. Synthetic Data Generation  
9. Visualization  
10. Planned Features  
11. Contributing  
12. Citation  
13. License

---

# 1. Project Motivation

Magnetotransport measurements are a fundamental probe of electronic properties in condensed matter systems. Typical experiments measure:

- Longitudinal resistivity (ρxx)
- Hall resistivity (ρxy)

as functions of magnetic field and temperature.

From these measurements researchers extract:

- Carrier densities
- Carrier mobilities
- Weak localization effects
- Quantum oscillations
- Fermi surface properties

However, experimental data analysis often requires **custom scripts**, making results difficult to reproduce or extend.

This project aims to provide:

- A **modular analysis pipeline**
- Standardized **fitting routines**
- Tools for **data visualization**
- **Synthetic data generation** for algorithm validation
- Extensible architecture for future experimental techniques

---

# 2. Features

Current and planned features include:

### Transport Modeling
- Two-band transport model fitting
- Carrier density extraction
- Mobility estimation

### Quantum Corrections
- Weak localization / anti-localization analysis
- Hikami-Larkin-Nagaoka (HLN) fitting

### Quantum Oscillations
- Shubnikov–de Haas oscillation detection
- Fourier transform analysis
- Frequency extraction

### Data Processing
- Noise filtering
- Background subtraction
- Interpolation and resampling

### Visualization
- Magnetoresistance curves
- Hall resistivity plots
- Oscillation spectra
- Fitting diagnostics

### Synthetic Data Generation
Synthetic datasets are used to:

- Validate algorithms
- Benchmark fitting routines
- Test robustness to noise

These datasets simulate realistic magnetotransport experiments.

---

# 3. Physics Background

## 3.1 Two-Band Transport Model

In many materials both electrons and holes contribute to transport.

The conductivity tensor is given by:

σxx = e Σ (ni μi / (1 + (μi B)^2))

σxy = e Σ (ni μi^2 B / (1 + (μi B)^2))

The resistivity tensor is obtained by matrix inversion:

ρ = σ⁻¹

Fitting these equations allows extraction of:

- Carrier densities (n)
- Mobilities (μ)

---

## 3.2 Weak Localization (HLN Model)

Quantum interference in diffusive systems produces a characteristic low-field magnetoconductance correction.

The Hikami–Larkin–Nagaoka formula is

Δσ(B) = −α (e² / (2π² ħ)) [ ψ(1/2 + Bφ / B) − ln(Bφ / B) ]

where

α = prefactor related to number of channels  
Bφ = phase coherence field

Fitting this model allows extraction of:

- Phase coherence length
- Spin-orbit interaction strength

---

## 3.3 Shubnikov–de Haas Oscillations

At high magnetic fields, Landau quantization produces oscillations in resistivity:

ρxx ∝ cos(2πF / B + φ)

where

F = oscillation frequency related to Fermi surface area.

Fourier analysis of these oscillations provides:

- Fermi surface cross sections
- Effective masses
- Quantum lifetimes

---

# 4. Repository Structure


qtransport/
│
├── app.py
│ Streamlit interface
│
├── core/
│ ├── data_model.py
│ TransportDataset class
│
│ ├── statistics.py
│ Statistical analysis utilities
│
│ ├── preprocessing.py
│ Data filtering and smoothing
│
│ ├── fitting.py
│ Model fitting algorithms
│
│ └── visualization.py
│ Plotting and analysis tools
│
├── models/
│ ├── two_band.py
│ Two-band transport model
│
│ ├── hln.py
│ Weak localization model
│
│ └── sdh.py
│ Quantum oscillation analysis
│
├── examples/
│ synthetic_data_generator.py
│ Generate test datasets
│
├── tests/
│ Unit tests and validation scripts
│
└── README.md


---

# 5. Installation

Clone the repository

```bash
git clone https://github.com/Sherlock-191b/qtransport.git

cd magnetotransport-toolkit
```

Create a virtual environment

```bash
python -m venv venv
```

Activate the environment

Linux / Mac

```bash
source venv/bin/activate
```

Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Typical dependencies include


numpy
scipy
pandas
matplotlib
streamlit


---

# 6. Usage

Run the interactive interface

```bash
streamlit run app.py
```

This launches a browser interface where you can:

- Upload experimental data
- Select analysis models
- Fit transport parameters
- Visualize results

---

# 7. Example Workflow

### Step 1: Load Data

Input dataset format


B_field,rho_xx,rho_xy,temperature
-10,0.0021,0.0004,2
-9.9,0.0020,0.00039,2
...


### Step 2: Preprocess Data

Typical preprocessing steps:

- Remove outliers
- Smooth noise
- Normalize datasets

### Step 3: Choose Model

Available models include

- Two-band transport
- Weak localization
- Quantum oscillations

### Step 4: Fit Parameters

The toolkit performs nonlinear optimization to determine:

- Carrier density
- Mobility
- Phase coherence field
- Oscillation frequencies

### Step 5: Visualize Results

Plots include:

- Raw vs fitted curves
- Residual analysis
- Fourier spectra

---

# 8. Synthetic Data Generation

Synthetic datasets can be generated for testing.

Run:
```bash

cd examples
python synthetic_data_generator.py
```

Available experiment types:

1. Two-band transport
2. Weak localization
3. SdH oscillations

These datasets include:

- realistic parameter ranges
- Gaussian noise
- metadata headers

Example output:

experiment = two_band
n_e = 1.3e22
mu_e = 0.5
temperature = 2

B_field,rho_xx,rho_xy
-10,0.0021,0.0005
...


Synthetic datasets are intended **only for algorithm validation** before applying the toolkit to real experimental data.

---

# 9. Visualization

Visualization utilities include:

- Magnetoresistance plots
- Hall curves
- Residual plots
- Fourier spectra of oscillations

Example plots include:

- ρxx vs B
- ρxy vs B
- oscillation amplitude vs 1/B

---

# 10. Planned Features

Future development will include:

### Advanced Transport Models

- Multi-band transport
- Anomalous Hall effect
- Topological transport signatures

### Improved Analysis

- Bayesian parameter estimation
- Uncertainty propagation
- Automatic peak detection

### Expanded Data Support

- Temperature dependent transport
- Angle dependent magnetoresistance
- Multi-terminal measurements

### Performance Improvements

- Parallel fitting algorithms
- GPU accelerated FFT routines

---

# 11. Contributing

Contributions are welcome.

Potential areas include:

- new transport models
- improved fitting algorithms
- experimental dataset support
- visualization improvements

Steps to contribute

1. Fork the repository
2. Create a new branch
3. Implement changes
4. Submit a pull request

---

# 12. Citation

If you use this toolkit in research, please cite:


Magnetotransport Data Analysis Toolkit
Sanju S Pillai
GitHub Repository (in development)


---

# 13. License

This project will be released under the MIT License.

---

# Author

Sanju S Pillai  
Physics Graduate Researcher  
Interested in condensed matter physics, stochastic dynamics, and computational modeling.

---

# Disclaimer

This toolkit is currently **under active development** and should be considered experimental.](https://github.com/Sherlock-191b/qtranspor)
