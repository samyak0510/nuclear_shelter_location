# AI-Optimized Nuclear Shelter Siting (UFLP)

## Overview
This project applies Genetic Algorithm (GA) optimization to identify optimal geographic locations for nuclear shelters across the United States. The problem is framed as an Uncapacitated Facility Location Problem (UFLP). The objective is to maximize population coverage while ensuring shelters are positioned outside high-risk blast zones (yield-scaled exclusion radius) and near necessary infrastructure.

## Team Members
- Murtaza Nipplewala
- Aartika Parmar
- Samyak Shah

## Project Structure
```
nuclear_shelter_location/
│
├── data/
│   ├── raw/               # Raw downloaded data (census, targets, urban areas)
│   └── processed/         # Cleaned CSVs/GeoJSONs
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py     # Functions to load Census, Urban Area, Target data
│   ├── fitness.py         # Fitness function (Coverage, Safety, Access)
│   ├── genetic_algo.py    # GA Class (Select, Crossover, Mutate)
│   ├── baseline.py        # Greedy heuristic implementation
│   ├── utils.py           # Helpers (distance calc, plotting)
│   └── main.py            # Entry point
│
├── notebooks/             # Jupyter notebooks for exploration
│
├── pyproject.toml         # Project packaging and dependencies
├── .gitignore
├── README.md
└── report/                # Drafts of your final paper
```

- `src/`: Core Python modules.
  - `data_loader.py`: Handles Census, Urban Area, Nuclear Target, and OSM data ingestion; includes preprocessing (null removal, deduplication, normalization).
  - `fitness.py`: Calculates coverage, safety constraints, and accessibility scores.
  - `genetic_algo.py`: Implements the GA (Selection, Crossover, Mutation).
  - `baseline.py`: Implements the Greedy Heuristic for comparison.
  - `main.py`: Entry point for execution.
- `data/`: Storage for raw and processed datasets.
- `notebooks/`: Exploratory data analysis and visualization.
- `report/`: Drafts and final version of the project paper.

## Data Card

### 1. U.S. Population by Zip Code (Census)

| Field | Description |
|---|---|
| **Source** | U.S. Census Bureau |
| **Download** | [Kaggle – US Population by Zip Code](https://www.kaggle.com/datasets/census/us-population-by-zip-code) |
| **File** | `population_by_zip_2010.csv` |
| **Purpose** | Provides zip-code-level population counts used as demand nodes for shelter coverage calculations. |

| Column | Description |
|---|---|
| `zip_code` | 5-digit USPS ZIP Code |
| `latitude` | Latitude of the ZIP Code centroid |
| `longitude` | Longitude of the ZIP Code centroid |
| `population` | Total estimated population within the ZIP Code |
| `state` | U.S. state the ZIP Code belongs to |

### 2. U.S. Nuclear Targets Database

| Field | Description |
|---|---|
| **Source** | Nuclear War Map (Christopher Minson LLC) |
| **Download** | [Nuclear War Map – Target List](https://www.nuclearwarmap.com/targetlist.html) |
| **File** | `us_nuclear_targets.xlsx` |
| **Purpose** | Lists potential nuclear strike targets; used to define yield-scaled blast exclusion radii around each target. ZIP Codes within these radii are excluded as candidate shelter sites. |

| Column | Description |
|---|---|
| `State` | U.S. state where the target is located |
| `Target` | Name of the target (e.g., city, military base, airport) |
| `Category` | Target classification: Military, Economic, Government, or Transportation |
| `Lat` | Latitude of the target |
| `Lng` | Longitude of the target |
| `Yield` | Estimated warhead yield (e.g., 500kt, 1000kt) |
| `Type` | Detonation type: Air Burst or Surface Burst |

### 3. United States Urban Areas Dataset

| Field | Description |
|---|---|
| **Source** | The Devastator (Kaggle) |
| **Download** | [Kaggle – United States Urban Areas Dataset](https://www.kaggle.com/datasets/thedevastator/united-states-urban-areas-dataset?resource=download) |
| **File** | `Urban_Areas.csv` |
| **Purpose** | Provides population density and geographic extent of U.S. urban areas; used for infrastructure accessibility scoring and population density weighting. |

| Column | Description |
|---|---|
| `Name` | Name of the urban area |
| `State` | U.S. state(s) the urban area spans |
| `Land Area (sq mi)` | Total land area in square miles |
| `Population` | Total population of the urban area |
| `Population Density` | People per square mile |
| `Latitude` | Latitude of the urban area centroid |
| `Longitude` | Longitude of the urban area centroid |

### 4. Optional Road Network Data (Not used in current pipeline)

| Field | Description |
|---|---|
| **Source** | OpenStreetMap via the OSMnx Python library |
| **Download** | Fetched programmatically using `osmnx` (no static download link) |
| **Purpose** | Potential future extension. The current implementation does **not** use OSM road graph features in preprocessing or fitness scoring. |

| Attribute | Description |
|---|---|
| `osmid` | Unique OpenStreetMap feature ID |
| `highway` | Road classification (motorway, primary, secondary, etc.) |
| `geometry` | Shapely LineString geometry of the road segment |
| `length` | Length of the road segment in meters |

## Brief Explanation of Code
The system encodes candidate solutions as binary vectors where each bit represents a US Zip Code (1 = Shelter, 0 = No Shelter).
1. **Initialization**: Generates random populations, masking out unsafe zones using yield-scaled blast radii around targets.
2. **Fitness Function**: Evaluates solutions based on total population covered within a service radius, rewards proximity to urban infrastructure, and penalizes overly expensive solutions (too many shelters).
3. **Evolution**: Uses tournament selection, uniform crossover, and bit-flip mutation to evolve solutions over generations.
4. **Evaluation**: Compares final GA results against a Greedy Baseline heuristic.

## Tools and Dependencies
This project is built in **Python 3.11+** and packaged with **setuptools** via `pyproject.toml`. Key libraries include:
- `numpy`, `pandas`: Data manipulation.
- `scipy`: Sparse matrix acceleration.
- `optuna`: Bayesian hyperparameter optimization.
- `pgeocode`: ZIP code geocoding for lat/lon enrichment.
- `openpyxl`: Excel loader for target data.
- `matplotlib`: Visualization.

See `pyproject.toml` for the full list of dependencies.

## Setup and Run

### 1. Environment Setup
```bash
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the project and all dependencies
pip install -e .
```

### 2. Prepare Data
Place the raw files in the `data/` directory:
- `population_by_zip_2010.csv`
- `us_nuclear_targets.xlsx`
- `Urban_Areas.csv`

### 3. Run the Code
```bash
python src/main.py
```

### 4. View Results
The results are saved in the `results/` directory, including:
- `convergence_plot.png`
- `comparison_plot.png` (GA vs Greedy visualization)
- `shelter_map.png`
- `ga_vs_greedy_metrics.csv`
- `final_results.json`
- `convergence_data.csv`
