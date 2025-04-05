# A2ACMOP and INSGA-III Implementation

This project implements the Air-to-Air Communication Multi-Objective Optimization Problem (A2ACMOP) and the Improved Non-Dominated Sorting Genetic Algorithm III (INSGA-III) as described in the provided paper, using the DEAP library.

## Project Structure

- `main.py`: Main script to run the optimization.
- `problem.py`: Defines the A2ACMOP problem, including objective functions and constraints.
- `algorithm.py`: Implements the INSGA-III algorithm and its custom operators (OBL, DBC, DBM, ALO, BH).
- `visualization.py`: Contains functions for visualizing the results (Pareto front, convergence, deployment).
- `config.py`: Stores configuration parameters for the problem and algorithm.
- `requirements.txt`: Lists project dependencies.
- `README.md`: This file.

## Setup

1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`

## Usage

Run the main script:

```bash
python main.py
```

The script will perform the optimization and generate visualizations if enabled in `config.py`.

## Notes

- The energy model in `problem.py` is currently simplified using placeholder factors. For accurate results, replace it with the detailed model from reference [5] using the appropriate constants (P_B, P_I, v_tip, v_0, d_0, rho, s, A, m, g).
- Some algorithm parameters (e.g., operator probabilities/counts, BH factor) might require tuning based on the paper or further experimentation. 