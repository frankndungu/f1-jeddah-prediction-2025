import fastf1
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
from matplotlib.patches import Rectangle
warnings.filterwarnings('ignore')

# Setup
cache_dir = 'f1_cache'
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

# Define 2025 race results from the provided data
# Australian GP
australian_gp_results_2025 = [
    {"position": 1, "driver": "Lando Norris", "nationality": "United Kingdom", "team": "McLaren", "number": 4, "time": "1:42:06.304", "points": 25},
    {"position": 2, "driver": "Max Verstappen", "nationality": "Netherlands", "team": "Red Bull", "number": 1, "time": "+0.895s", "points": 18},
    {"position": 3, "driver": "George Russell", "nationality": "United Kingdom", "team": "Mercedes", "number": 63, "time": "+8.481s", "points": 15},
    {"position": 4, "driver": "Andrea Kimi Antonelli", "nationality": "Italy", "team": "Mercedes", "number": 12, "time": "+10.135s", "points": 12},
    {"position": 5, "driver": "Alexander Albon", "nationality": "Thailand", "team": "Williams", "number": 23, "time": "+12.773s", "points": 10},
    {"position": 6, "driver": "Lance Stroll", "nationality": "Canada", "team": "Aston Martin", "number": 18, "time": "+17.413s", "points": 8},
    {"position": 7, "driver": "Nico Hülkenberg", "nationality": "Germany", "team": "Kick Sauber", "number": 27, "time": "+18.423s", "points": 6},
    {"position": 8, "driver": "Charles Leclerc", "nationality": "Monaco", "team": "Ferrari", "number": 16, "time": "+19.826s", "points": 4},
    {"position": 9, "driver": "Oscar Piastri", "nationality": "Australia", "team": "McLaren", "number": 81, "time": "+20.448s", "points": 2},
    {"position": 10, "driver": "Lewis Hamilton", "nationality": "United Kingdom", "team": "Ferrari", "number": 44, "time": "+22.473s", "points": 1},
    {"position": 11, "driver": "Pierre Gasly", "nationality": "France", "team": "Alpine", "number": 10, "time": "+26.502s", "points": 0},
    {"position": 12, "driver": "Yuki Tsunoda", "nationality": "Japan", "team": "RB", "number": 22, "time": "+29.884s", "points": 0},
    {"position": 13, "driver": "Esteban Ocon", "nationality": "France", "team": "Haas", "number": 31, "time": "+33.161s", "points": 0},
    {"position": 14, "driver": "Oliver Bearman", "nationality": "United Kingdom", "team": "Haas", "number": 87, "time": "+40.351s", "points": 0},
    {"position": 15, "driver": "Liam Lawson", "nationality": "New Zealand", "team": "Red Bull", "number": 30, "time": "DNF", "points": 0},
    {"position": 16, "driver": "Gabriel Bortoleto", "nationality": "Brazil", "team": "Kick Sauber", "number": 5, "time": "DNF", "points": 0},
    {"position": 17, "driver": "Fernando Alonso", "nationality": "Spain", "team": "Aston Martin", "number": 14, "time": "DNF", "points": 0},
    {"position": 18, "driver": "Carlos Sainz Jr.", "nationality": "Spain", "team": "Williams", "number": 55, "time": "DNF", "points": 0},
    {"position": 19, "driver": "Jack Doohan", "nationality": "Australia", "team": "Alpine", "number": 7, "time": "DNF", "points": 0},
    {"position": 20, "driver": "Isack Hadjar", "nationality": "France", "team": "RB", "number": 6, "time": "DNF", "points": 0},
]

# Shanghai GP
shanghai_gp_results_2025 = [
    {"position": 1, "driver": "Oscar Piastri", "nationality": "Australia", "team": "McLaren", "number": 81, "time": "1:30:55.026", "points": 25},
    {"position": 2, "driver": "Lando Norris", "nationality": "United Kingdom", "team": "McLaren", "number": 4, "time": "+9.748s", "points": 18},
    {"position": 3, "driver": "George Russell", "nationality": "United Kingdom", "team": "Mercedes", "number": 63, "time": "+11.097s", "points": 15},
    {"position": 4, "driver": "Max Verstappen", "nationality": "Netherlands", "team": "Red Bull", "number": 1, "time": "+16.656s", "points": 12},
    {"position": 5, "driver": "Charles Leclerc", "nationality": "Monaco", "team": "Ferrari", "number": 16, "time": "+56 Laps", "points": 0},
    {"position": 6, "driver": "Lewis Hamilton", "nationality": "United Kingdom", "team": "Ferrari", "number": 44, "time": "+56 Laps", "points": 0},
    {"position": 7, "driver": "Esteban Ocon", "nationality": "France", "team": "Haas", "number": 31, "time": "+49.969s", "points": 10},
    {"position": 8, "driver": "Andrea Kimi Antonelli", "nationality": "Italy", "team": "Mercedes", "number": 12, "time": "+53.748s", "points": 8},
    {"position": 9, "driver": "Alexander Albon", "nationality": "Thailand", "team": "Williams", "number": 23, "time": "+56.321s", "points": 6},
    {"position": 10, "driver": "Oliver Bearman", "nationality": "United Kingdom", "team": "Haas", "number": 87, "time": "+61.303s", "points": 4},
    {"position": 11, "driver": "Pierre Gasly", "nationality": "France", "team": "Alpine", "number": 10, "time": "+56 Laps", "points": 0},
    {"position": 12, "driver": "Lance Stroll", "nationality": "Canada", "team": "Aston Martin", "number": 18, "time": "+70.204s", "points": 2},
    {"position": 13, "driver": "Carlos Sainz Jr.", "nationality": "Spain", "team": "Williams", "number": 55, "time": "+76.387s", "points": 1},
    {"position": 14, "driver": "Isack Hadjar", "nationality": "France", "team": "RB", "number": 6, "time": "+78.875s", "points": 0},
    {"position": 15, "driver": "Jack Doohan", "nationality": "Australia", "team": "Alpine", "number": 7, "time": "+88.401s", "points": 0},
    {"position": 16, "driver": "Liam Lawson", "nationality": "New Zealand", "team": "Red Bull", "number": 30, "time": "+81.147s", "points": 0},
    {"position": 17, "driver": "Gabriel Bortoleto", "nationality": "Brazil", "team": "Kick Sauber", "number": 5, "time": "+1 Lap", "points": 0},
    {"position": 18, "driver": "Nico Hülkenberg", "nationality": "Germany", "team": "Kick Sauber", "number": 27, "time": "+1 Lap", "points": 0},
    {"position": 19, "driver": "Yuki Tsunoda", "nationality": "Japan", "team": "RB", "number": 22, "time": "+1 Lap", "points": 0},
    {"position": 20, "driver": "Fernando Alonso", "nationality": "Spain", "team": "Aston Martin", "number": 14, "time": "DNF", "points": 0},
]

# Suzuka GP
suzuka_gp_results_2025 = [
    {"position": 1, "driver": "Max Verstappen", "nationality": "Netherlands", "team": "Red Bull", "number": 1, "time": "1:22:06.983", "points": 25},
    {"position": 2, "driver": "Lando Norris", "nationality": "United Kingdom", "team": "McLaren", "number": 4, "time": "+1.423s", "points": 18},
    {"position": 3, "driver": "Oscar Piastri", "nationality": "Australia", "team": "McLaren", "number": 81, "time": "+2.129s", "points": 15},
    {"position": 4, "driver": "Charles Leclerc", "nationality": "Monaco", "team": "Ferrari", "number": 16, "time": "+16.097s", "points": 12},
    {"position": 5, "driver": "George Russell", "nationality": "United Kingdom", "team": "Mercedes", "number": 63, "time": "+17.362s", "points": 10},
    {"position": 6, "driver": "Andrea Kimi Antonelli", "nationality": "Italy", "team": "Mercedes", "number": 12, "time": "+18.671s", "points": 8},
    {"position": 7, "driver": "Lewis Hamilton", "nationality": "United Kingdom", "team": "Ferrari", "number": 44, "time": "+29.182s", "points": 6},
    {"position": 8, "driver": "Isack Hadjar", "nationality": "France", "team": "RB", "number": 6, "time": "+37.134s", "points": 4},
    {"position": 9, "driver": "Alexander Albon", "nationality": "Thailand", "team": "Williams", "number": 23, "time": "+40.367s", "points": 2},
    {"position": 10, "driver": "Oliver Bearman", "nationality": "United Kingdom", "team": "Haas", "number": 87, "time": "+54.529s", "points": 1},
    {"position": 11, "driver": "Fernando Alonso", "nationality": "Spain", "team": "Aston Martin", "number": 14, "time": "+57.333s", "points": 0},
    {"position": 12, "driver": "Yuki Tsunoda", "nationality": "Japan", "team": "Red Bull", "number": 22, "time": "+58.401s", "points": 0},
    {"position": 13, "driver": "Pierre Gasly", "nationality": "France", "team": "Alpine", "number": 10, "time": "+62.122s", "points": 0},
    {"position": 14, "driver": "Carlos Sainz Jr.", "nationality": "Spain", "team": "Williams", "number": 55, "time": "+74.129s", "points": 0},
    {"position": 15, "driver": "Jack Doohan", "nationality": "Australia", "team": "Alpine", "number": 7, "time": "+81.314s", "points": 0},
    {"position": 16, "driver": "Nico Hülkenberg", "nationality": "Germany", "team": "Kick Sauber", "number": 27, "time": "+81.957s", "points": 0},
    {"position": 17, "driver": "Liam Lawson", "nationality": "New Zealand", "team": "RB", "number": 30, "time": "+82.734s", "points": 0},
    {"position": 18, "driver": "Esteban Ocon", "nationality": "France", "team": "Haas", "number": 31, "time": "+83.438s", "points": 0},
    {"position": 19, "driver": "Gabriel Bortoleto", "nationality": "Brazil", "team": "Kick Sauber", "number": 5, "time": "+83.897s", "points": 0},
    {"position": 20, "driver": "Lance Stroll", "nationality": "Canada", "team": "Aston Martin", "number": 18, "time": "+1 Lap", "points": 0},
]

# Bahrain GP
bahrain_gp_results_2025 = [
    {"position": 1, "driver": "Oscar Piastri", "nationality": "Australia", "team": "McLaren", "number": 81, "time": "1:35:39.435", "points": 25},
    {"position": 2, "driver": "George Russell", "nationality": "United Kingdom", "team": "Mercedes", "number": 63, "time": "+15.499s", "points": 18},
    {"position": 3, "driver": "Lando Norris", "nationality": "United Kingdom", "team": "McLaren", "number": 4, "time": "+16.273s", "points": 15},
    {"position": 4, "driver": "Charles Leclerc", "nationality": "Monaco", "team": "Ferrari", "number": 16, "time": "+19.679s", "points": 12},
    {"position": 5, "driver": "Lewis Hamilton", "nationality": "United Kingdom", "team": "Ferrari", "number": 44, "time": "+27.993s", "points": 10},
    {"position": 6, "driver": "Max Verstappen", "nationality": "Netherlands", "team": "Red Bull", "number": 1, "time": "+34.395s", "points": 8},
    {"position": 7, "driver": "Pierre Gasly", "nationality": "France", "team": "Alpine", "number": 10, "time": "+36.002s", "points": 6},
    {"position": 8, "driver": "Esteban Ocon", "nationality": "France", "team": "Haas", "number": 31, "time": "+44.244s", "points": 4},
    {"position": 9, "driver": "Yuki Tsunoda", "nationality": "Japan", "team": "Red Bull", "number": 22, "time": "+45.061s", "points": 2},
    {"position": 10, "driver": "Oliver Bearman", "nationality": "United Kingdom", "team": "Haas", "number": 87, "time": "+47.594s", "points": 1},
    {"position": 11, "driver": "Andrea Kimi Antonelli", "nationality": "Italy", "team": "Mercedes", "number": 12, "time": "+48.016s", "points": 0},
    {"position": 12, "driver": "Alexander Albon", "nationality": "Thailand", "team": "Williams", "number": 23, "time": "+48.839s", "points": 0},
    {"position": 13, "driver": "Isack Hadjar", "nationality": "France", "team": "RB", "number": 6, "time": "+56.314s", "points": 0},
    {"position": 14, "driver": "Jack Doohan", "nationality": "Australia", "team": "Alpine", "number": 7, "time": "+57.806s", "points": 0},
    {"position": 15, "driver": "Fernando Alonso", "nationality": "Spain", "team": "Aston Martin", "number": 14, "time": "+60.340s", "points": 0},
    {"position": 16, "driver": "Liam Lawson", "nationality": "New Zealand", "team": "RB", "number": 30, "time": "+64.435s", "points": 0},
    {"position": 17, "driver": "Lance Stroll", "nationality": "Canada", "team": "Aston Martin", "number": 18, "time": "+65.489s", "points": 0},
    {"position": 18, "driver": "Gabriel Bortoleto", "nationality": "Brazil", "team": "Kick Sauber", "number": 5, "time": "+66.872s", "points": 0},
    {"position": 19, "driver": "Carlos Sainz Jr.", "nationality": "Spain", "team": "Williams", "number": 55, "time": "DNF", "points": 0},
    {"position": 20, "driver": "Nico Hülkenberg", "nationality": "Germany", "team": "Kick Sauber", "number": 27, "time": "DNF", "points": 0},
]

# Create a list of all active drivers from 2025 data
all_2025_races = [australian_gp_results_2025, shanghai_gp_results_2025, suzuka_gp_results_2025, bahrain_gp_results_2025]
all_drivers = set()
for race in all_2025_races:
    for driver in race:
        all_drivers.add(driver["driver"])

# Create a mapping of driver numbers
driver_mapping = {}
for race in all_2025_races:
    for entry in race:
        driver_mapping[entry['driver']] = entry['number']

# Convert 2025 race data to pandas DataFrames
race_dfs = []
for i, race_data in enumerate(all_2025_races):
    race_name = ["Australian GP", "Shanghai GP", "Suzuka GP", "Bahrain GP"][i]
    race_df = pd.DataFrame(race_data)
    race_df['Race'] = race_name
    race_df['Season'] = 2025
    race_dfs.append(race_df)

recent_form_df = pd.concat(race_dfs)

# Load historical Saudi Arabian GP data
saudi_gp_data = []
for season in [2022, 2023, 2024]:
    try:
        # Find the Saudi Arabian GP in each season
        season_calendar = fastf1.get_event_schedule(season)
        saudi_events = season_calendar[season_calendar['EventName'].str.contains('Saudi|Jeddah', case=False)]
        
        if not saudi_events.empty:
            saudi_event = saudi_events.iloc[0]
            session = fastf1.get_session(season, saudi_event['EventName'], 'R')
            session.load()
            results = session.results[['DriverNumber', 'Position', 'Points', 'GridPosition', 'Team']]
            results['Season'] = season
            results['Circuit'] = 'Jeddah'
            saudi_gp_data.append(results)
            print(f"Loaded Saudi Arabian GP data for {season}")
        else:
            print(f"No Saudi Arabian GP found for {season}")
            
    except Exception as e:
        print(f"Error loading Saudi Arabian GP data for {season}: {e}")
        continue

# Combine historical data
if saudi_gp_data:
    historical_saudi_df = pd.concat(saudi_gp_data)
    historical_saudi_df['DriverNumber'] = historical_saudi_df['DriverNumber'].astype(int)
    historical_saudi_df['Position'] = pd.to_numeric(historical_saudi_df['Position'], errors='coerce').fillna(25)
    historical_saudi_df['GridPosition'] = pd.to_numeric(historical_saudi_df['GridPosition'], errors='coerce').fillna(25)
else:
    # Create a dummy dataframe if no historical data could be loaded
    print("Warning: No historical Saudi Arabian GP data could be loaded. Using dummy data.")
    historical_saudi_df = pd.DataFrame({
        'DriverNumber': [],
        'Position': [],
        'Points': [],
        'GridPosition': [],
        'Team': [],
        'Season': [],
        'Circuit': []
    })

# Calculate form metrics from 2025 race data
driver_stats = {}

for driver in all_drivers:
    driver_results = recent_form_df[recent_form_df['driver'] == driver]
    if not driver_results.empty:
        avg_position = driver_results['position'].mean()
        avg_points = driver_results['points'].mean()
        finishes = len(driver_results)
        team = driver_results.iloc[-1]['team']  # Get most recent team
        
        # Track position changes from grid to finish
        # For this, we'd need qualifying data which we don't have in the 2025 race data
        # Instead, we'll calculate race-to-race position changes
        pos_changes = []
        race_positions = driver_results.sort_values('Race')['position'].tolist()
        if len(race_positions) > 1:
            for i in range(1, len(race_positions)):
                pos_changes.append(race_positions[i-1] - race_positions[i])
        
        avg_pos_change = np.mean(pos_changes) if pos_changes else 0
        
        driver_stats[driver] = {
            'AvgPosition': avg_position,
            'AvgPoints': avg_points,
            'Races': finishes,
            'Team': team,
            'DriverNumber': driver_mapping.get(driver, 0),
            'AvgPosChange': avg_pos_change,
            'LastRacePosition': driver_results.iloc[-1]['position'] if not driver_results.empty else 20
        }
    else:
        # Default values for drivers with no recent data
        driver_stats[driver] = {
            'AvgPosition': 15,
            'AvgPoints': 0,
            'Races': 0,
            'Team': "Unknown",
            'DriverNumber': driver_mapping.get(driver, 0),
            'AvgPosChange': 0,
            'LastRacePosition': 20
        }

# Create a DataFrame from driver stats
drivers_df = pd.DataFrame.from_dict(driver_stats, orient='index').reset_index()
drivers_df.rename(columns={'index': 'FullName'}, inplace=True)

# Add Saudi-specific performance based on historical data
for i, driver in drivers_df.iterrows():
    driver_number = driver['DriverNumber']
    
    # Get this driver's historical performance at Saudi tracks
    driver_saudi_data = historical_saudi_df[historical_saudi_df['DriverNumber'] == driver_number]
    
    if not driver_saudi_data.empty:
        # Calculate average position delta (grid to finish) at Saudi
        pos_deltas = driver_saudi_data['GridPosition'] - driver_saudi_data['Position']
        avg_saudi_delta = pos_deltas.mean()
        
        # Calculate average finish position at Saudi
        avg_saudi_position = driver_saudi_data['Position'].mean()
        
        drivers_df.at[i, 'SaudiAvgDelta'] = avg_saudi_delta
        drivers_df.at[i, 'SaudiAvgPosition'] = avg_saudi_position
        drivers_df.at[i, 'SaudiExperience'] = len(driver_saudi_data)
    else:
        # No Saudi data for this driver
        drivers_df.at[i, 'SaudiAvgDelta'] = 0
        drivers_df.at[i, 'SaudiAvgPosition'] = driver['AvgPosition']  # Use season average as fallback
        drivers_df.at[i, 'SaudiExperience'] = 0

# Standardize team names
team_mapping = {
    'McLaren': 'McLaren',
    'Red Bull': 'Red Bull Racing',
    'Mercedes': 'Mercedes',
    'Ferrari': 'Ferrari',
    'Aston Martin': 'Aston Martin',
    'Williams': 'Williams',
    'Alpine': 'Alpine',
    'Haas': 'Haas F1 Team',
    'Kick Sauber': 'Kick Sauber',
    'RB': 'VCARB'
}

drivers_df['Team'] = drivers_df['Team'].map(lambda x: team_mapping.get(x, x))

# Create team color mapping
team_colors = {
    'McLaren': '#FF8700',          # Orange
    'Red Bull Racing': '#0600EF',  # Dark blue
    'Red Bull': '#0600EF',         # Dark blue
    'Ferrari': '#DC0000',          # Red
    'Mercedes': '#00D2BE',         # Turquoise
    'Aston Martin': '#006F62',     # British racing green
    'Williams': '#005AFF',         # Blue
    'Alpine': '#0090FF',           # Blue
    'Haas F1 Team': '#888888',     # Gray
    'Haas': '#888888',             # Gray
    'Kick Sauber': '#900000',      # Burgundy
    'Sauber': '#900000',           # Burgundy
    'VCARB': '#2B4562',            # Navy blue
    'Racing Bulls': '#2B4562'      # Navy blue
}

# Assign colors to each driver based on team
drivers_df['Color'] = drivers_df['Team'].map(team_colors)

# Estimate qualifying positions based on recent form
# Since we're not using actual qualifying data, we'll make an estimate
# based on recent race performance with some randomization
np.random.seed(42)  # For reproducible results

# Start with the last race position and add some randomness
qualifying_noise = np.random.normal(0, 2, size=len(drivers_df))
estimated_quali = drivers_df['LastRacePosition'] + qualifying_noise

# Make sure positions are valid (1-20) and unique
estimated_quali = np.clip(estimated_quali, 1, 20)
estimated_quali_ranks = pd.Series(estimated_quali).rank(method='first').astype(int)

# Update drivers_df with estimated qualifying positions
drivers_df['GridPosition'] = estimated_quali_ranks

# Calculate predicted positions for Saudi Arabian GP
feature_data = []
for i, driver in drivers_df.iterrows():
    driver_name = driver['FullName']
    team = driver['Team']
    grid_pos = driver['GridPosition']
    
    # Base prediction on combination of:
    # 1. Recent form (70%)
    # 2. Saudi-specific performance (30%)
    form_component = driver['AvgPosition'] * 0.7
    
    # Use Saudi-specific data if available, otherwise just use form
    if driver['SaudiExperience'] > 0:
        saudi_component = driver['SaudiAvgPosition'] * 0.3
    else:
        # No Saudi experience - rely more on recent form and add slight penalty for unknown track
        saudi_component = driver['AvgPosition'] * 0.3 * 1.05  # 5% penalty
    
    base_prediction = form_component + saudi_component
    
    # Adjust prediction based on grid position and recent position change tendency
    grid_adjustment = -0.2 * driver['AvgPosChange']  # If driver tends to gain positions, this is positive
    grid_factor = 0.2  # How much grid position affects the prediction
    
    adjusted_prediction = base_prediction + (grid_pos * grid_factor) + grid_adjustment
    
    # Add uncertainty - higher for midfield, lower for top teams
    if driver['AvgPosition'] <= 5:
        uncertainty = 1.5  # Low uncertainty for top teams
    elif driver['AvgPosition'] <= 10:
        uncertainty = 2.0  # Medium uncertainty for midfield
    else:
        uncertainty = 2.5  # High uncertainty for backmarkers
    
    # Special track-specific adjustments
    # Saudi Arabia is a high-speed circuit with walls - experienced drivers have an advantage
    experience_factor = 1.0
    if driver['SaudiExperience'] > 1:
        experience_factor = 0.95  # 5% advantage for Saudi experience
    elif driver['Races'] > 3:
        experience_factor = 0.98  # 2% advantage for experienced drivers
    
    # Adjust prediction with experience factor
    final_prediction = adjusted_prediction * experience_factor
    
    # Add to features
    feature_data.append({
        'DriverNumber': driver['DriverNumber'],
        'FullName': driver_name,
        'Team': team,
        'GridPosition': grid_pos,
        'Experience': driver['Races'],
        'SaudiExperience': driver['SaudiExperience'],
        'RecentAvgPos': driver['AvgPosition'],
        'RecentAvgPoints': driver['AvgPoints'],
        'PredictedPosition': final_prediction,
        'Uncertainty': uncertainty,
        'Color': driver['Color']
    })

# Create DataFrame with all features
df_prediction = pd.DataFrame(feature_data)

# Run simulation for race outcome
sim_count = 1000
all_results = []

for sim in range(sim_count):
    sim_results = df_prediction.copy()
    
    # Add random noise based on uncertainty
    sim_results['SimPosition'] = sim_results.apply(
        lambda x: max(1, x['PredictedPosition'] + np.random.normal(0, x['Uncertainty'])), 
        axis=1
    )
    
    # Factor in race incidents - Saudi has a high chance of safety cars and crashes
    # About 15% chance of DNF or significant problem (higher than usual)
    incident_mask = np.random.random(len(sim_results)) < 0.15
    sim_results.loc[incident_mask, 'SimPosition'] += np.random.randint(5, 15, size=sum(incident_mask))
    
    # Add first lap chaos factor - front positions are safer but Saudi has tight first corner
    first_lap_chaos = np.random.normal(0, 3.5, size=len(sim_results)) * (sim_results['GridPosition'] / 10)
    sim_results['SimPosition'] += first_lap_chaos
    
    # Sort by simulated position for this race
    race_result = sim_results.sort_values('SimPosition').reset_index(drop=True)
    race_result['SimFinish'] = race_result.index + 1
    
    all_results.append(race_result[['DriverNumber', 'FullName', 'SimFinish']])

# Aggregate results from all simulations
final_results = pd.concat(all_results)
avg_positions = final_results.groupby(['DriverNumber', 'FullName'])['SimFinish'].agg(['mean', 'std', 'min', 'max'])
avg_positions = avg_positions.reset_index()

# Join with driver and team info
final_df = pd.merge(avg_positions, df_prediction[['DriverNumber', 'FullName', 'Team', 'GridPosition', 'Color']], 
                   on=['DriverNumber', 'FullName'])

# Calculate probability of podium finish
podium_counts = {}
for sim_df in all_results:
    for driver in sim_df.loc[sim_df['SimFinish'] <= 3, 'FullName'].unique():
        podium_counts[driver] = podium_counts.get(driver, 0) + 1

# Convert to DataFrame and calculate percentages
podium_probs = pd.DataFrame(list(podium_counts.items()), columns=['FullName', 'PodiumCount'])
podium_probs['PodiumProbability'] = podium_probs['PodiumCount'] / sim_count * 100

# Merge with final results
final_df = pd.merge(final_df, podium_probs[['FullName', 'PodiumProbability']], 
                   on='FullName', how='left')
final_df['PodiumProbability'] = final_df['PodiumProbability'].fillna(0)

# Calculate points expectations
points_mapping = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}

# Calculate expected points for each driver
expected_points = {}
for sim_df in all_results:
    for _, row in sim_df.iterrows():
        driver = row['FullName']
        position = row['SimFinish']
        points = points_mapping.get(position, 0)
        expected_points[driver] = expected_points.get(driver, 0) + points

# Convert to average expected points
for driver in expected_points:
    expected_points[driver] /= sim_count

# Add to final DataFrame
expected_points_df = pd.DataFrame(list(expected_points.items()), columns=['FullName', 'ExpectedPoints'])
final_df = pd.merge(final_df, expected_points_df, on='FullName', how='left')

# Sort by expected finish position
final_df = final_df.sort_values('mean').reset_index(drop=True)
final_df['ExpectedPosition'] = final_df.index + 1

# Visualize the results
plt.figure(figsize=(12, 10))
plt.style.use('ggplot')

# Set up position ranges (error bars)
yerr = np.zeros((2, len(final_df)))
yerr[0] = final_df['mean'] - final_df['min']  # bottom error
yerr[1] = final_df['max'] - final_df['mean']  # top error

# Plot expected positions with error bars showing range
bars = plt.barh(final_df['FullName'], final_df['mean'], 
                xerr=yerr,
                alpha=0.7,
                color=final_df['Color'])

# Add grid position indicators
for i, (_, row) in enumerate(final_df.iterrows()):
    plt.plot(row['GridPosition'], i, 'o', color='black', markersize=6)
    
# Add probability percentages
for i, (_, row) in enumerate(final_df.iterrows()):
    plt.text(0.5, i, f"{row['PodiumProbability']:.1f}% Pod", 
             va='center', ha='left', fontsize=8, color='black')
    plt.text(row['mean'] + 0.2, i, f"{row['ExpectedPoints']:.1f} Pts", 
             va='center', ha='left', fontsize=8, color='black')

# Customize the plot
plt.title('2025 Saudi Arabian GP Prediction', fontsize=16)
plt.xlabel('Position', fontsize=12)
plt.ylabel('Driver', fontsize=12)
plt.xlim(0, 21)  # Position range from 1 to 20
plt.gca().invert_xaxis()  # Invert x-axis so 1st position is on the right
plt.grid(True, axis='x')

# Add a legend for grid positions
plt.plot([], [], 'o', color='black', label='Grid Position')

# Customize ticks
plt.xticks(range(1, 21))

plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig('saudi_gp_prediction.png', dpi=300)

# Create team performance plot
plt.figure(figsize=(12, 8))

# Group by team to get team stats
team_stats = final_df.groupby('Team').agg({
    'mean': ['mean', 'min'],
    'ExpectedPoints': 'sum'
}).reset_index()

team_stats.columns = ['Team', 'AvgPosition', 'BestPosition', 'TotalExpectedPoints']
team_stats = team_stats.sort_values('TotalExpectedPoints', ascending=False)

# Get team colors
team_stats['Color'] = team_stats['Team'].map(team_colors)

# Create bar plot for expected team points
plt.barh(team_stats['Team'], team_stats['TotalExpectedPoints'], color=team_stats['Color'], alpha=0.8)

# Add labels
for i, row in team_stats.iterrows():
    plt.text(row['TotalExpectedPoints'] + 1, i, f"Avg Pos: {row['AvgPosition']:.1f}, Best: {row['BestPosition']:.0f}", 
             va='center', fontsize=10)

plt.title('2025 Saudi Arabian GP - Expected Team Performance', fontsize=16)
plt.xlabel('Expected Points', fontsize=12)
plt.ylabel('Team', fontsize=12)
plt.grid(True, axis='x')
plt.tight_layout()
plt.savefig('saudi_gp_team_prediction.png', dpi=300)

# Create a heatmap showing the distribution of predicted finishing positions
finish_distribution = pd.pivot_table(
    final_results, 
    index='FullName', 
    columns='SimFinish', 
    aggfunc='size', 
    fill_value=0
)

# Calculate percentage of simulations for each position
for col in finish_distribution.columns:
    finish_distribution[col] = finish_distribution[col] / sim_count * 100

# Get driver order based on expected position
driver_order = final_df['FullName'].tolist()
finish_distribution = finish_distribution.reindex(driver_order)

plt.figure(figsize=(14, 10))
sns.heatmap(finish_distribution, cmap='YlOrRd', annot=False, fmt='.1f', 
            linewidths=0.5, cbar_kws={'label': '% of Simulations'})

plt.title('Predicted Finishing Position Distribution - Saudi Arabian GP 2025', fontsize=16)
plt.xlabel('Finishing Position', fontsize=12)
plt.ylabel('Driver', fontsize=12)
plt.tight_layout()
plt.savefig('saudi_gp_position_heatmap.png', dpi=300)

# Create head-to-head comparison for specific drivers
def plot_driver_comparison(driver1, driver2):
    driver1_results = final_results[final_results['FullName'] == driver1]['SimFinish']
    driver2_results = final_results[final_results['FullName'] == driver2]['SimFinish']
    
    plt.figure(figsize=(12, 6))
    
    # Get driver colors
    driver1_color = final_df[final_df['FullName'] == driver1]['Color'].iloc[0]
    driver2_color = final_df[final_df['FullName'] == driver2]['Color'].iloc[0]
    
    # Plot histograms
    plt.hist(driver1_results, bins=range(1, 22), alpha=0.7, color=driver1_color, label=driver1)
    plt.hist(driver2_results, bins=range(1, 22), alpha=0.7, color=driver2_color, label=driver2)
    
    # Calculate how often driver1 finishes ahead
    ahead_count = sum(d1 < d2 for d1, d2 in zip(
        final_results[final_results['FullName'] == driver1]['SimFinish'],
        final_results[final_results['FullName'] == driver2]['SimFinish']
    ))
    ahead_pct = ahead_count / sim_count * 100
    
    plt.title(f'{driver1} vs {driver2} - Saudi Arabian GP 2025\n{driver1} finishes ahead in {ahead_pct:.1f}% of simulations', fontsize=14)
    plt.xlabel('Finishing Position', fontsize=12)
    plt.ylabel('Number of Simulations', fontsize=12)
    plt.xticks(range(1, 21))
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{driver1.split()[1].lower()}_vs_{driver2.split()[1].lower()}.png', dpi=300)

# Compare top drivers
plot_driver_comparison('Max Verstappen', 'Lando Norris')
plot_driver_comparison('Oscar Piastri', 'George Russell')

# Calculate probabilities for different scenarios
points_finishes = {}
for sim_df in all_results:
    for driver in sim_df.loc[sim_df['SimFinish'] <= 10, 'FullName'].unique():
        points_finishes[driver] = points_finishes.get(driver, 0) + 1

points_probs = pd.DataFrame(list(points_finishes.items()), columns=['FullName', 'PointsCount'])
points_probs['PointsProbability'] = points_probs['PointsCount'] / sim_count * 100

# Merge with final results
final_df = pd.merge(final_df, points_probs[['FullName', 'PointsProbability']], 
                   on='FullName', how='left')
final_df['PointsProbability'] = final_df['PointsProbability'].fillna(0)

# Calculate win probabilities
win_counts = {}
for sim_df in all_results:
    for driver in sim_df.loc[sim_df['SimFinish'] == 1, 'FullName'].unique():
        win_counts[driver] = win_counts.get(driver, 0) + 1

win_probs = pd.DataFrame(list(win_counts.items()), columns=['FullName', 'WinCount'])
win_probs['WinProbability'] = win_probs['WinCount'] / sim_count * 100

# Merge with final results
final_df = pd.merge(final_df, win_probs[['FullName', 'WinProbability']], 
                   on='FullName', how='left')
final_df['WinProbability'] = final_df['WinProbability'].fillna(0)

# Print detailed prediction results
print("\n2025 Saudi Arabian Grand Prix Prediction")
print("="*50)
print(f"Based on {sim_count} race simulations\n")

print(final_df[['ExpectedPosition', 'FullName', 'Team', 'GridPosition', 
                'mean', 'min', 'max', 'WinProbability', 'PodiumProbability', 
                'PointsProbability', 'ExpectedPoints']].to_string(index=False))

# Create a horizontal bar chart for win probabilities
plt.figure(figsize=(12, 8))
win_data = final_df[final_df['WinProbability'] > 0].sort_values('WinProbability', ascending=True)

plt.barh(win_data['FullName'], win_data['WinProbability'], 
         color=win_data['Color'], alpha=0.8)

plt.title('2025 Saudi Arabian GP - Win Probability', fontsize=16)
plt.xlabel('Win Probability (%)', fontsize=12)
plt.ylabel('Driver', fontsize=12)
plt.grid(True, axis='x')
plt.tight_layout()
plt.savefig('saudi_gp_win_probability.png', dpi=300)

# Create a plot showing expected points vs grid position
plt.figure(figsize=(12, 8))

scatter = plt.scatter(final_df['GridPosition'], final_df['ExpectedPoints'], 
                      c=final_df['Color'], s=100, alpha=0.7)

# Add driver labels
for i, row in final_df.iterrows():
    plt.text(row['GridPosition']+0.1, row['ExpectedPoints']+0.2, 
             row['FullName'].split()[-1], fontsize=10)
    
# Add trend line
z = np.polyfit(final_df['GridPosition'], final_df['ExpectedPoints'], 1)
p = np.poly1d(z)
x_trend = np.linspace(1, 20, 100)
plt.plot(x_trend, p(x_trend), "r--", alpha=0.7)

plt.title('Grid Position vs Expected Points - Saudi Arabian GP 2025', fontsize=16)
plt.xlabel('Grid Position', fontsize=12)
plt.ylabel('Expected Points', fontsize=12)
plt.grid(True)
plt.tight_layout()
plt.savefig('grid_vs_points.png', dpi=300)

# Show all plots
plt.show()

# Export predictions to CSV
final_df.to_csv('saudi_gp_predictions.csv', index=False)

print("\nAnalysis complete! All visualizations have been saved.")