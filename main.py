# Placeholder for main execution logic

import random
import numpy as np
import time
from tqdm import tqdm
import multiprocessing
import os
import json
import pickle

from deap import base, creator, tools, algorithms

# Configuration parameters
from config import (
    POP_SIZE, MAX_GEN, NUM_OBJECTIVES, CXPB, MUTPB, ALOPB, USE_OBL,
    IND_SIZE, LOWER_BOUNDS, UPPER_BOUNDS, REC_A_IDX, REC_B_IDX,
    PLOT_PARETO_FRONT, PLOT_CONVERGENCE, PLOT_DEPLOYMENT, DEPLOYMENT_SNAPSHOT_GEN,
    NUM_DIVISIONS, USE_IMPROVED_ALGORITHM
)

# Problem definition and evaluation functions
from problem import evaluate_a2acmop, set_initial_positions, check_constraints, INITIAL_POSITIONS_A, INITIAL_POSITIONS_B

# Custom algorithm operators
from algorithm import (
    generate_opposition, crossover_dbc, mutate_dbm, update_alo, apply_bh_operator,
    get_bounds, check_bounds
)

# Visualization functions
from visualization import plot_pareto_front, plot_convergence, plot_deployment


# If using improved algorithm, import related functionality
if USE_IMPROVED_ALGORITHM:
    from improved_algorithm import (
        get_adaptive_reference_points, generate_variational_offspring,
        evaluate_with_surrogate, update_surrogate_with_new_data
    )

# --- DEAP Setup ---

# Create fitness class (minimize three objectives: -RateAB, -RateBA, Energy)
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0, 1.0))

# Create individual class (list with fitness attribute)
creator.create("Individual", list, fitness=creator.FitnessMin)

# --- Toolbox Registration ---
toolbox = base.Toolbox()

# Attribute generator (generates random float for individual)
def attr_item(low, up):
    return random.uniform(low, up)

toolbox.register("attr_float", attr_item)

# Individual generator (repeatedly calls attr_float)
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_float for _ in range(IND_SIZE)), n=1)

# Adjust individual generation to ensure it's within bounds
# (applying boundaries directly during generation seems simpler)
def individual_bounded():
    """Create a random individual within bounds"""
    low, up = get_bounds()
    ind = creator.Individual(random.uniform(l, u) for l, u in zip(low, up))
    # Ensure receiver indices are approximately integers initially, for easier processing
    ind[REC_A_IDX] = random.randint(low[REC_A_IDX], up[REC_A_IDX])  # Receiver index in group B
    ind[REC_B_IDX] = random.randint(low[REC_B_IDX], up[REC_B_IDX])  # Receiver index in group A
    return ind
toolbox.register("individual_bounded", individual_bounded)

# Population generator
toolbox.register("population", tools.initRepeat, list, toolbox.individual_bounded)

# OBL initialization
def population_obl(n):
    pop = toolbox.population(n=n // 2 if USE_OBL else n)
    if USE_OBL:
        opp_pop = []
        lb, ub = get_bounds()
        for ind in pop:
            opp_ind = generate_opposition(ind, lb, ub)
            opp_pop.append(opp_ind)
        # DEAP algorithms typically do first selection after initial evaluation, so no filtering here
        # Filtering logic can be done before the main loop starts, or let NSGA3 selection handle it
        pop.extend(opp_pop)
    return pop
toolbox.register("population_obl", population_obl)

# Register evaluation function
toolbox.register("evaluate", evaluate_a2acmop)

# If using improved algorithm, register surrogate model evaluation function
if USE_IMPROVED_ALGORITHM:
    toolbox.register("evaluate_surrogate", evaluate_with_surrogate, real_eval_func=evaluate_a2acmop)
    # Set current generation attribute for variational operator (initially 0)
    toolbox.current_gen = 0
    toolbox.max_gen = MAX_GEN

# Register custom crossover operator (DBC)
# Note: cxUniform's indpb controls probability of each gene being exchanged
toolbox.register("mate", crossover_dbc, indpb_continuous=0.5) # Continuous part gene exchange probability

# Register custom mutation operator (DBM)
# eta is the crowding parameter for mutPolynomialBounded
# indpb is the probability of each gene being mutated
lb, ub = get_bounds()
toolbox.register("mutate", mutate_dbm, low=lb, up=ub, mu=20.0, indpb_continuous=1.0/IND_SIZE)

# Register custom ALO update operator
toolbox.register("update_alo", update_alo)

# Register custom BH operator
toolbox.register("apply_bh", apply_bh_operator)

# Generate reference points for NSGA-III
ref_points = tools.uniform_reference_points(nobj=NUM_OBJECTIVES, p=NUM_DIVISIONS)

# Register selection operator (NSGA-III)
toolbox.register("select", tools.selNSGA3, ref_points=ref_points)

# Save ref_points as toolbox attribute for later updates
toolbox.ref_points = ref_points

# Register boundary check decorator
toolbox.decorate("mate", check_bounds(lb, ub))
toolbox.decorate("mutate", check_bounds(lb, ub))
toolbox.decorate("update_alo", check_bounds(lb, ub))
toolbox.decorate("apply_bh", check_bounds(lb, ub))

# Define process pool initialization function at module level
# Moved outside main() to enable pickling
from problem import set_positions_from_arrays
def init_worker(init_pos_a, init_pos_b):
    set_positions_from_arrays(init_pos_a, init_pos_b)

# --- Main Program ---
def main():
    random.seed(42) # Set random seed for reproducibility
    np.random.seed(42)

    # Create result save directory
    result_dir = "improv" if USE_IMPROVED_ALGORITHM else "origin"
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(f"{result_dir}/plots", exist_ok=True)  # Create subdirectory for visualization results

    # Set UAV initial positions (before creating process pool)
    print("Generating UAV initial positions...")
    init_pos_a, init_pos_b = set_initial_positions()
    
    # Setup multiprocessing
    num_processes = 3  # Use all CPU cores except one
    num_processes = max(1, num_processes)  # Ensure at least one core is used
    print(f"Using {num_processes} CPU cores for parallel computation...")
    
    # Replace default map function with parallel version
    # Use maxtasksperchild to limit the number of tasks each process handles, preventing memory leaks
    pool = multiprocessing.Pool(processes=num_processes, initializer=init_worker, initargs=(init_pos_a, init_pos_b), maxtasksperchild=50)
    toolbox.register("map", pool.map)

    # Setup statistics
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    logbook = tools.Logbook()
    logbook.header = "gen", "evals", "std", "min", "avg", "max"

    # Create initial population (including OBL)
    print(f"Initializing population (size: {POP_SIZE})...")
    pop = toolbox.population_obl(n=POP_SIZE)

    # Evaluate invalid individuals in the initial population
    invalid_ind = [ind for ind in pop if not ind.fitness.valid]
    
    # If using improved surrogate model
    if USE_IMPROVED_ALGORITHM and hasattr(toolbox, 'evaluate_surrogate'):
        fitnesses = [toolbox.evaluate_surrogate(ind) for ind in invalid_ind]
    else:
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        
    for ind, fit in tqdm(zip(invalid_ind, fitnesses), total=len(invalid_ind), desc="Initial population evaluation [0/5]"):
        ind.fitness.values = fit

    # If using improved algorithm, update surrogate model
    if USE_IMPROVED_ALGORITHM:
        update_surrogate_with_new_data(invalid_ind, [ind.fitness.values for ind in invalid_ind], 0)

    # If OBL was used, initial population size might be POP_SIZE*2, need first selection
    if USE_OBL:
        print("Applying initial NSGA-III selection (due to using OBL)...")
        pop = toolbox.select(pop, POP_SIZE)

    # Record initial statistics
    record = stats.compile(pop)
    logbook.record(gen=0, evals=len(invalid_ind), **record)
    print(logbook.stream)

    # Create text log file
    with open(f"{result_dir}/optimization_log.txt", "w", encoding="utf-8") as log_file:
        log_file.write(f"Optimization Algorithm: {'Improved NSGA-III' if USE_IMPROVED_ALGORITHM else 'Original NSGA-III'}\n")
        log_file.write(f"Population Size: {POP_SIZE}, Maximum Generations: {MAX_GEN}\n")
        log_file.write(f"Crossover Probability: {CXPB}, Mutation Probability: {MUTPB}, ALO Probability: {ALOPB}\n")
        log_file.write(f"Using OBL Initialization: {'Yes' if USE_OBL else 'No'}\n\n")
        log_file.write("=== Optimization Log ===\n")
        log_file.write(f"Generation 0: {logbook[0]}\n")

    # Plot initial deployment (select an initial best individual)
    if PLOT_DEPLOYMENT and 0 in DEPLOYMENT_SNAPSHOT_GEN:
        # Ensure initial positions are set
        if INITIAL_POSITIONS_A is None or INITIAL_POSITIONS_B is None:
            print("Initial positions not set, cannot plot initial deployment")
        else:
            # Select a point from the Pareto front (e.g., lowest energy)
            best_ind_initial = min(pop, key=lambda x: x.fitness.values[2])
            plot_deployment(best_ind_initial, 0, result_dir=result_dir)

    # Start evolution
    print("Starting evolution...")
    start_time = time.time()
    for gen in tqdm(range(1, MAX_GEN + 1), desc=f"Evolution process [1/5]"):
        # Set current generation (for improved algorithm)
        if USE_IMPROVED_ALGORITHM:
            toolbox.current_gen = gen
            toolbox.max_gen = MAX_GEN

        # --- Generate offspring ---
        # Use standard tournament selection instead of DCD variant (DCD not suitable for NSGA-III)
        selected_parents = tools.selTournament(pop, len(pop), tournsize=3)
        offspring = [toolbox.clone(ind) for ind in selected_parents]

        # If using improved algorithm, generate additional offspring with variational distribution sampling
        if USE_IMPROVED_ALGORITHM:
            # Update reference points (if needed)
            if hasattr(toolbox, 'ref_points'):
                toolbox.ref_points = get_adaptive_reference_points(pop, gen, toolbox.ref_points)
                # Re-register selection operator
                toolbox.register("select", tools.selNSGA3, ref_points=toolbox.ref_points)
            
            # Generate variational distribution sampled offspring
            var_offspring = generate_variational_offspring(pop, toolbox)
            if var_offspring:
                offspring.extend(var_offspring)

        # Apply crossover (DBC)
        for i in tqdm(range(0, len(offspring), 2), desc=f"Crossover operation [2/5]", leave=False):
            if random.random() < CXPB and i+1 < len(offspring):
                 toolbox.mate(offspring[i], offspring[i+1])
                 del offspring[i].fitness.values # Invalidate fitness after crossover
                 del offspring[i+1].fitness.values

        # Apply mutation (DBM)
        for i in tqdm(range(len(offspring)), desc=f"Mutation operation [3/5]", leave=False):
            if random.random() < MUTPB:
                toolbox.mutate(offspring[i])
                del offspring[i].fitness.values # Invalidate fitness after mutation

        # Apply ALO
        if ALOPB > 0 and len(pop) > 0:
             elite = tools.selBest(pop, 1)[0] # Approximate elite
             for i in tqdm(range(len(offspring)), desc=f"ALO operation [4/5]", leave=False):
                 if random.random() < ALOPB:
                    antlion = random.choice(pop) # Randomly select antlion
                    toolbox.update_alo(offspring[i], antlion, elite, MAX_GEN, gen)
                    del offspring[i].fitness.values # Invalidate fitness after ALO update

        # --- Combine parents and offspring ---
        combined_pop = pop + offspring

        # --- Apply BH operator to combined population --- (Algo 4.7, line 7)
        # BH application probability or frequency can be configured, simplified here to apply to all individuals
        for i in tqdm(range(len(combined_pop)), desc=f"BH operator application [5/5]", leave=False):
            toolbox.apply_bh(combined_pop[i], gen, MAX_GEN)
            # BH changes position, theoretically needs re-evaluation
            del combined_pop[i].fitness.values

        # --- Evaluate all individuals with invalid fitness ---
        # (after crossover, mutation, ALO, BH)
        invalid_ind = [ind for ind in combined_pop if not ind.fitness.valid]
        
        # If using improved surrogate model
        if USE_IMPROVED_ALGORITHM and hasattr(toolbox, 'evaluate_surrogate'):
            # Use surrogate model to evaluate
            surrogate_fitnesses = []
            true_eval_inds = []
            
            for ind in invalid_ind:
                fitness = evaluate_with_surrogate(ind, toolbox.evaluate)
                surrogate_fitnesses.append(fitness)
                if not hasattr(ind, '_surrogate_evaluated') or not ind._surrogate_evaluated:
                    true_eval_inds.append(ind)
            
            # Update surrogate model
            if true_eval_inds:
                update_surrogate_with_new_data(true_eval_inds, 
                                              [ind.fitness.values for ind in true_eval_inds], 
                                              gen)
                
            # Set fitness
            for ind, fit in zip(invalid_ind, surrogate_fitnesses):
                ind.fitness.values = fit
                ind._surrogate_evaluated = True
        else:
            # Use real evaluation function
            fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
            for ind, fit in tqdm(zip(invalid_ind, fitnesses), total=len(invalid_ind), desc="Fitness evaluation", leave=False):
                ind.fitness.values = fit

        # --- NSGA-III selection ---
        pop = toolbox.select(combined_pop, POP_SIZE)

        # --- Record statistics ---
        record = stats.compile(pop)
        logbook.record(gen=gen, evals=len(invalid_ind), **record)
        print(logbook.stream)
        
        # Record to text log
        with open(f"{result_dir}/optimization_log.txt", "a", encoding="utf-8") as log_file:
            log_file.write(f"Generation {gen}: {logbook[gen]}\n")

        # --- Visualization snapshots ---
        if PLOT_DEPLOYMENT and gen in DEPLOYMENT_SNAPSHOT_GEN:
            # Select the individual with lowest energy from current Pareto front for plotting
            current_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
            if current_front:
                 best_ind_snapshot = min(current_front, key=lambda x: x.fitness.values[2])
                 plot_deployment(best_ind_snapshot, gen, result_dir=result_dir)
            else:
                 print(f"No non-dominated solutions found in generation {gen} for deployment plot.")

        if PLOT_PARETO_FRONT and gen % 50 == 0: # Plot Pareto front every 50 generations
            current_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
            if current_front:
                 plot_pareto_front(current_front, gen, result_dir=result_dir)

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Evolution completed, time taken: {execution_time:.2f} seconds")

    # --- Final results and visualization ---
    print("\nFinal population Pareto front:")
    final_front = tools.sortNondominated(pop, len(pop), first_front_only=True)[0]
    
    # Save final Pareto front results to text file
    with open(f"{result_dir}/final_results.txt", "w", encoding="utf-8") as results_file:
        results_file.write(f"Optimization Algorithm: {'Improved NSGA-III' if USE_IMPROVED_ALGORITHM else 'Original NSGA-III'}\n")
        results_file.write(f"Total Time: {execution_time:.2f} seconds\n\n")
        results_file.write("=== Final Pareto Front ===\n")
        results_file.write("ID\tRate_AB (bps)\tRate_BA (bps)\tEnergy (J)\n")
        
        # Sort individuals in the front by energy
        sorted_front = sorted(final_front, key=lambda x: x.fitness.values[2])
        
        for i, ind in enumerate(sorted_front):
            rate_ab = -ind.fitness.values[0]  # Convert back to original value
            rate_ba = -ind.fitness.values[1]  # Convert back to original value
            energy = ind.fitness.values[2]
            results_file.write(f"{i}\t{rate_ab:.4e}\t{rate_ba:.4e}\t{energy:.4e}\n")
            print(f"  Individual {i}: {rate_ab:.2e} (R_AB), {rate_ba:.2e} (R_BA), {energy:.2e} (E_tot)")
    
    # Save Pareto front data for subsequent analysis
    pareto_data = {
        'algorithm': 'improved' if USE_IMPROVED_ALGORITHM else 'original',
        'execution_time': execution_time,
        'objectives': [(ind.fitness.values[0], ind.fitness.values[1], ind.fitness.values[2]) 
                       for ind in final_front],
        'hypervolume': None,  # Will be calculated in metric.py
        'spread': None        # Will be calculated in metric.py
    }
    
    with open(f"{result_dir}/pareto_data.pkl", "wb") as f:
        pickle.dump(pareto_data, f)

    # Save convergence history data
    convergence_data = {
        'generations': logbook.select('gen'),
        'avg_values': logbook.select('avg'),
        'min_values': logbook.select('min'),
        'max_values': logbook.select('max'),
    }
    
    with open(f"{result_dir}/convergence_data.pkl", "wb") as f:
        pickle.dump(convergence_data, f)

    if PLOT_PARETO_FRONT:
        plot_pareto_front(final_front, gen=-1, result_dir=result_dir) # Plot final Pareto front

    if PLOT_CONVERGENCE:
        plot_convergence(logbook, result_dir=result_dir)

    if PLOT_DEPLOYMENT and MAX_GEN not in DEPLOYMENT_SNAPSHOT_GEN:
        # Plot deployment of the best individual from final generation
        best_ind_final = min(final_front, key=lambda x: x.fitness.values[2]) # Select by lowest energy
        plot_deployment(best_ind_final, MAX_GEN, result_dir=result_dir)

    return pop, logbook

if __name__ == "__main__":
    multiprocessing.set_start_method('spawn')
    try:
        final_pop, stats_log = main()
        # More analysis of final_pop can be added here
        print("\nOptimization process completed.")
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")
    finally:
        # Close all process pools
        if 'pool' in locals():
            pool.close()
            pool.join()
            print("All process pools closed.")
