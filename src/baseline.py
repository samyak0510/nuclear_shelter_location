import numpy as np
import pandas as pd
from src.utils import haversine_distance

def greedy_heuristic(zip_codes, targets, infra_scores, max_shelters=100, service_radius=50):
    """
    Greedy Baseline Heuristic for UFLP.
    
    Logic:
    1. Filter out unsafe zones (within 15 miles of targets).
    2. Iteratively select the safe zip code that covers the most 
       *uncovered* population until max_shelters is reached.
    
    Args:
        zip_codes (pd.DataFrame): Census data with lat, lon, population.
        targets (list): List of urban target dicts with lat, lon.
        infra_scores (np.array): Infrastructure score per zip code.
        max_shelters (int): Budget constraint (number of shelters to build).
        service_radius (int): Miles within which a shelter covers a population.
        
    Returns:
        tuple: (chromosome_binary_array, final_fitness_score)
    """
    print(f"Running Greedy Baseline (k={max_shelters})...")
    
    n = len(zip_codes)
    chromosome = np.zeros(n, dtype=int)
    
    # 1. Identify Safe Zones (Hard Constraint)
    # Re-implementing safety check here for independence, 
    # though you could import from fitness.py
    safe_mask = np.ones(n, dtype=bool)
    for i in range(n):
        z_lat, z_lon = zip_codes.iloc[i]['lat'], zip_codes.iloc[i]['lon']
        for target in targets:
            dist = haversine_distance(z_lat, z_lon, target['lat'], target['lon'])
            if dist < 15: # 15-mile exclusion
                safe_mask[i] = False
                break
    
    # Get indices of safe zip codes
    safe_indices = np.where(safe_mask)[0]
    
    # 2. Greedy Selection Loop
    # Track which population centers are already covered
    covered_population = np.zeros(n, dtype=bool)
    selected_indices = []
    
    # Pre-calculate distances matrix for safe zones only (Optimization for speed)
    # Note: For 30k zip codes, this matrix is large. 
    # For the starter code, we compute on the fly or limit safe_indices.
    
    for k in range(max_shelters):
        best_idx = -1
        best_gain = -1
        
        # Find the safe zip code that covers the most currently UNCOVERED population
        for idx in safe_indices:
            if idx in selected_indices:
                continue
                
            s_lat = zip_codes.iloc[idx]['lat']
            s_lon = zip_codes.iloc[idx]['lon']
            
            # Calculate marginal gain (new population covered by this shelter)
            current_gain = 0
            for j in range(n):
                if covered_population[j]:
                    continue # Already covered
                
                # Check if this new shelter covers zip code j
                dist = haversine_distance(s_lat, s_lon, 
                                          zip_codes.iloc[j]['lat'], 
                                          zip_codes.iloc[j]['lon'])
                if dist <= service_radius:
                    current_gain += zip_codes.iloc[j]['population']
            
            # Add infrastructure score as a tie-breaker/small bonus
            current_gain += infra_scores[idx] * 100 
            
            if current_gain > best_gain:
                best_gain = current_gain
                best_idx = idx
        
        if best_idx != -1:
            selected_indices.append(best_idx)
            chromosome[best_idx] = 1
            
            # Mark population as covered by this new shelter
            s_lat = zip_codes.iloc[best_idx]['lat']
            s_lon = zip_codes.iloc[best_idx]['lon']
            for j in range(n):
                dist = haversine_distance(s_lat, s_lon, 
                                          zip_codes.iloc[j]['lat'], 
                                          zip_codes.iloc[j]['lon'])
                if dist <= service_radius:
                    covered_population[j] = True
        else:
            print("No more valid shelters found.")
            break
            
    # 3. Calculate Final Fitness using the same logic as GA
    # (Importing FitnessFunction class would be better, but duplicating for standalone clarity)
    total_covered_pop = np.sum(zip_codes['population'].values[covered_population])
    avg_infra = np.mean(infra_scores[selected_indices]) if selected_indices else 0
    final_score = (total_covered_pop / 1000) + (avg_infra * 100)
    
    print(f"Greedy Baseline Complete. Selected {len(selected_indices)} shelters.")
    print(f"Greedy Fitness Score: {final_score}")
    
    return chromosome, final_score