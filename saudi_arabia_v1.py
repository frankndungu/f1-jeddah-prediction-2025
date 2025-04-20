import fastf1
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import os
from matplotlib.patches import Rectangle, Circle
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

# Add Saudi Arabia qualifying data
saudi_arabia_qualifying_2025 = [
    {"position": 1, "driver": "Max Verstappen", "nationality": "Netherlands", "team": "Red Bull", "q1": "1m27.778s", "q2": "1m27.529s", "q3": "1m27.294s"},
    {"position": 2, "driver": "Oscar Piastri", "nationality": "Australia", "team": "McLaren", "q1": "1m27.901s", "q2": "1m27.545s", "q3": "1m27.304s"},
    {"position": 3, "driver": "George Russell", "nationality": "United Kingdom", "team": "Mercedes", "q1": "1m28.282s", "q2": "1m27.599s", "q3": "1m27.407s"},
    {"position": 4, "driver": "Charles Leclerc", "nationality": "Monaco", "team": "Ferrari", "q1": "1m28.552s", "q2": "1m27.866s", "q3": "1m27.670s"},
    {"position": 5, "driver": "Kimi Antonelli", "nationality": "Italy", "team": "Mercedes", "q1": "1m28.128s", "q2": "1m27.798s", "q3": "1m27.866s"},
    {"position": 6, "driver": "Carlos Sainz", "nationality": "Spain", "team": "Williams", "q1": "1m28.354s", "q2": "1m28.024s", "q3": "1m28.164s"},
    {"position": 7, "driver": "Lewis Hamilton", "nationality": "United Kingdom", "team": "Ferrari", "q1": "1m28.372s", "q2": "1m28.102s", "q3": "1m28.201s"},
    {"position": 8, "driver": "Yuki Tsunoda", "nationality": "Japan", "team": "Red Bull", "q1": "1m28.226s", "q2": "1m27.990s", "q3": "1m28.204s"},
    {"position": 9, "driver": "Pierre Gasly", "nationality": "France", "team": "Alpine", "q1": "1m28.421s", "q2": "1m28.025s", "q3": "1m28.367s"},
    {"position": 10, "driver": "Lando Norris", "nationality": "United Kingdom", "team": "McLaren", "q1": "1m27.805s", "q2": "1m27.481s", "q3": "No Time Set"},
    {"position": 11, "driver": "Alex Albon", "nationality": "Thailand", "team": "Williams", "q1": "1m28.279s", "q2": "1m28.109s", "q3": None},
    {"position": 12, "driver": "Liam Lawson", "nationality": "New Zealand", "team": "RB", "q1": "1m28.561s", "q2": "1m28.191s", "q3": None},
    {"position": 13, "driver": "Fernando Alonso", "nationality": "Spain", "team": "Aston Martin", "q1": "1m28.548s", "q2": "1m28.303s", "q3": None},
    {"position": 14, "driver": "Isack Hadjar", "nationality": "France", "team": "RB", "q1": "1m28.571s", "q2": "1m28.418s", "q3": None},
    {"position": 15, "driver": "Oliver Bearman", "nationality": "United Kingdom", "team": "Haas", "q1": "1m28.536s", "q2": "1m28.646s", "q3": None},
    {"position": 16, "driver": "Lance Stroll", "nationality": "Canada", "team": "Aston Martin", "q1": "1m28.645s", "q2": None, "q3": None},
    {"position": 17, "driver": "Jack Doohan", "nationality": "Australia", "team": "Alpine", "q1": "1m28.739s", "q2": None, "q3": None},
    {"position": 18, "driver": "Nico Hulkenberg", "nationality": "Germany", "team": "Kick Sauber", "q1": "1m28.782s", "q2": None, "q3": None},
    {"position": 19, "driver": "Esteban Ocon", "nationality": "France", "team": "Haas", "q1": "1m29.092s", "q2": None, "q3": None},
    {"position": 20, "driver": "Gabriel Bortoleto", "nationality": "Brazil", "team": "Kick Sauber", "q1": "1m29.465s", "q2": None, "q3": None},
]

# Create a list of all active drivers from 2025 data
all_2025_races = [australian_gp_results_2025, shanghai_gp_results_2025, suzuka_gp_results_2025, bahrain_gp_results_2025]
all_drivers = set()
for race in all_2025_races:
    for driver in race:
        all_drivers.add(driver["driver"])

# Standardize driver names (handle Kimi Antonelli vs Andrea Kimi Antonelli)
driver_name_mapping = {
    "Kimi Antonelli": "Andrea Kimi Antonelli",
    "Alex Albon": "Alexander Albon",
    "Nico Hulkenberg": "Nico Hülkenberg"
}

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

# Now use the actual qualifying data for grid positions
grid_positions = {}
for entry in saudi_arabia_qualifying_2025:
    # Handle name differences
    driver_name = entry['driver']
    if driver_name in driver_name_mapping:
        driver_name = driver_name_mapping[driver_name]
    
    grid_positions[driver_name] = entry['position']

# Update the grid positions in the drivers DataFrame
drivers_df['GridPosition'] = drivers_df['FullName'].map(lambda x: grid_positions.get(x, 20))

# Calculate predicted positions for Saudi Arabian GP
feature_data = []
for i, driver in drivers_df.iterrows():
    driver_name = driver['FullName']
    team = driver['Team']
    
    # Create features for the model
    feature_row = {
        'DriverName': driver_name,
        'Team': team,
        'AvgPosition': driver['AvgPosition'],
        'AvgPoints': driver['AvgPoints'],
        'GridPosition': driver['GridPosition'],
        'SaudiAvgPosition': driver['SaudiAvgPosition'] if 'SaudiAvgPosition' in driver else driver['AvgPosition'],
        'SaudiAvgDelta': driver['SaudiAvgDelta'] if 'SaudiAvgDelta' in driver else 0,
        'SaudiExperience': driver['SaudiExperience'] if 'SaudiExperience' in driver else 0,
        'LastRacePosition': driver['LastRacePosition'],
        'AvgPosChange': driver['AvgPosChange'],
        'Color': driver['Color']
    }
    feature_data.append(feature_row)

# Create a DataFrame for prediction
prediction_df = pd.DataFrame(feature_data)

# Feature engineering: Create model features
X_features = prediction_df[['AvgPosition', 'AvgPoints', 'GridPosition', 'SaudiAvgPosition', 
                           'SaudiAvgDelta', 'SaudiExperience', 'LastRacePosition', 'AvgPosChange']]

# Use historical data to train a simple model (if we had target data)
# Since we don't have actual historical target data for training, we'll create a simple formula
# that predicts finish position based on a weighted combination of features
prediction_df['PredictedPosition'] = (
    0.3 * prediction_df['GridPosition'] +  # Grid position has strong influence
    0.3 * prediction_df['AvgPosition'] +   # Recent form is important
    0.2 * prediction_df['SaudiAvgPosition'] - # Track-specific performance matters
    0.1 * prediction_df['SaudiAvgDelta'] +  # Track-specific overtaking ability
    0.1 * prediction_df['AvgPosChange'] -   # General race pace vs qualifying
    0.05 * prediction_df['SaudiExperience']  # Experience at this track helps
)

# Estimate race pace and points
prediction_df['EstimatedRacePace'] = (
    0.5 * (21 - prediction_df['AvgPosition']) +
    0.3 * (21 - prediction_df['SaudiAvgPosition']) +
    0.2 * prediction_df['AvgPosChange']
)

# Sort by predicted position
prediction_df = prediction_df.sort_values('PredictedPosition')

# Round for display purposes
prediction_df['PredictedPosition'] = prediction_df['PredictedPosition'].round(1)

# Add position ranks
prediction_df['PredictedRank'] = range(1, len(prediction_df) + 1)

# Calculate points based on predicted positions (F1 points system)
points_map = {1: 25, 2: 18, 3: 15, 4: 12, 5: 10, 6: 8, 7: 6, 8: 4, 9: 2, 10: 1}
prediction_df['ExpectedPoints'] = prediction_df['PredictedRank'].map(lambda x: points_map.get(x, 0))

# Calculate position delta (positive means gained positions)
prediction_df['PositionDelta'] = prediction_df['GridPosition'] - prediction_df['PredictedPosition']

# Print prediction results
print("Predicted 2025 Saudi Arabian Grand Prix Results:")
print(prediction_df[['PredictedRank', 'DriverName', 'Team', 'GridPosition', 'PredictedPosition', 'ExpectedPoints']].to_string(index=False))


# ===== VISUALIZATION 1: MAIN PREDICTION CHART =====
plt.figure(figsize=(12, 8))
plt.style.use('dark_background')

# Create the bar chart
bars = plt.barh(prediction_df['DriverName'], prediction_df['PredictedRank'].max() + 1 - prediction_df['PredictedRank'], 
                color=prediction_df['Color'], alpha=0.8)

# Add grid position indicators
for i, (driver, grid, pred) in enumerate(zip(prediction_df['DriverName'], 
                                        prediction_df['GridPosition'], 
                                        prediction_df['PredictedRank'].max() + 1 - prediction_df['PredictedRank'])):
    plt.plot([0, pred], [i, i], 'w--', alpha=0.3)
    plt.text(0.2, i, f"P{int(grid)}", color='white', va='center', ha='left', fontsize=9)

# Customize the plot
plt.xlabel('Predicted Performance', fontsize=12)
plt.ylabel('')
plt.title('2025 Saudi Arabian Grand Prix Prediction', fontsize=16)
plt.gca().invert_yaxis()  # Invert y-axis to show best performer at the top

# Add the predicted position next to each bar
for i, v in enumerate(prediction_df['PredictedRank']):
    plt.text(prediction_df['PredictedRank'].max() + 1 - v + 0.2, i, f"P{v}", color='white', va='center', ha='left')

# Remove frame
plt.box(False)

# Set grid style
plt.grid(axis='x', linestyle='--', alpha=0.2)

# Add Otto Rentals caption
plt.figtext(0.5, 0.01, 
           "Powered by https://www.otto.rentals – Best and safest way to rent a car in Kenya", 
           ha='center', 
           fontsize=10, 
           color='#2E5EAA',
           weight='bold')

plt.tight_layout(pad=2.0)
plt.subplots_adjust(bottom=0.1)  # Make room for the caption

# Save the figure
plt.savefig('saudi_gp_prediction_2025.png', dpi=300, bbox_inches='tight')
plt.show()


# ===== VISUALIZATION 2: GRID VS PREDICTED POSITION =====
plt.figure(figsize=(10, 8))
plt.style.use('dark_background')

# Create scatter plot of grid vs predicted position
scatter = plt.scatter(prediction_df['GridPosition'], 
                     prediction_df['PredictedRank'], 
                     c=[team_colors.get(team, '#FFFFFF') for team in prediction_df['Team']], 
                     s=100, 
                     alpha=0.8)

# Add driver names as labels
for i, txt in enumerate(prediction_df['DriverName']):
    plt.annotate(txt.split(' ')[-1], # Use last name only
                (prediction_df['GridPosition'].iloc[i], prediction_df['PredictedRank'].iloc[i]),
                xytext=(5, 0), 
                textcoords='offset points',
                fontsize=9)

# Add diagonal line (no position change)
max_pos = max(prediction_df['GridPosition'].max(), prediction_df['PredictedRank'].max())
plt.plot([0, max_pos+1], [0, max_pos+1], 'w--', alpha=0.3)

# Areas for gainers and losers
plt.fill_between([0, max_pos+1], [0, 0], [max_pos+1, 0], color='green', alpha=0.1)
plt.fill_between([0, max_pos+1], [0, max_pos+1], [max_pos+1, max_pos+1], color='red', alpha=0.1)

# Add annotations for the areas
plt.text(max_pos/2, max_pos/4, 'GAINERS', color='lightgreen', ha='center', alpha=0.7)
plt.text(max_pos/2, max_pos*3/4, 'LOSERS', color='lightcoral', ha='center', alpha=0.7)

# Customize the plot
plt.xlabel('Grid Position', fontsize=12)
plt.ylabel('Predicted Finish Position', fontsize=12)
plt.title('Grid vs Predicted Finish - 2025 Saudi Arabian Grand Prix', fontsize=14)
plt.grid(linestyle='--', alpha=0.2)

# Set axis limits
plt.xlim(0, max_pos+1)
plt.ylim(0, max_pos+1)

# Invert y-axis (P1 at the top)
plt.gca().invert_yaxis()

# Add Otto Rentals caption
plt.figtext(0.5, 0.01, 
           "Powered by https://www.otto.rentals – Best and safest way to rent a car in Kenya", 
           ha='center', 
           fontsize=10, 
           color='#2E5EAA',
           weight='bold')

plt.tight_layout(pad=2.0)
plt.subplots_adjust(bottom=0.1)  # Make room for the caption

# Save the figure
plt.savefig('saudi_gp_grid_vs_prediction_2025.png', dpi=300, bbox_inches='tight')
plt.show()


# ===== VISUALIZATION 3: TEAM PERFORMANCE =====
team_prediction = prediction_df.groupby('Team').agg(
    AvgPredictedPosition=('PredictedPosition', 'mean'),
    BestPosition=('PredictedPosition', 'min'),
    ExpectedPoints=('ExpectedPoints', 'sum'),
    DriverCount=('DriverName', 'count')
).reset_index()

team_prediction = team_prediction.sort_values('AvgPredictedPosition')

# Add team colors
team_prediction['Color'] = team_prediction['Team'].map(team_colors)

plt.figure(figsize=(12, 8))
plt.style.use('dark_background')

# Create horizontal bar chart for team performance
bars = plt.barh(team_prediction['Team'], 
               team_prediction['ExpectedPoints'], 
               color=team_prediction['Color'], 
               alpha=0.8)

# Add team name and best position
for i, (team, avg_pos, best_pos, points) in enumerate(zip(team_prediction['Team'], 
                                                 team_prediction['AvgPredictedPosition'], 
                                                 team_prediction['BestPosition'],
                                                 team_prediction['ExpectedPoints'])):
    plt.text(points + 0.5, i, f"Avg: P{avg_pos:.1f}  Best: P{best_pos:.0f}  Points: {points:.0f}", 
             color='white', va='center', ha='left', fontsize=10)

# Customize the plot
plt.xlabel('Expected Points', fontsize=12)
plt.title('Team Performance Prediction - 2025 Saudi Arabian Grand Prix', fontsize=14)
plt.grid(axis='x', linestyle='--', alpha=0.2)
plt.box(False)

# Add Otto Rentals caption
plt.figtext(0.5, 0.01, 
           "Powered by https://www.otto.rentals – Best and safest way to rent a car in Kenya", 
           ha='center', 
           fontsize=10, 
           color='#2E5EAA',
           weight='bold')

plt.tight_layout(pad=2.0)
plt.subplots_adjust(bottom=0.1)  # Make room for the caption

# Save the figure
plt.savefig('saudi_gp_team_prediction_2025.png', dpi=300, bbox_inches='tight')
plt.show()


# ===== VISUALIZATION 4: DRIVERS TO WATCH =====
# Find drivers with big positive delta between grid and predicted finish
drivers_to_watch = prediction_df.sort_values('PositionDelta', ascending=False).head(5)

plt.figure(figsize=(12, 6))
plt.style.use('dark_background')

# Create bar chart for drivers to watch
bars = plt.bar(drivers_to_watch['DriverName'], 
              drivers_to_watch['PositionDelta'], 
              color=[drivers_to_watch['Color']], 
              alpha=0.8)

# Add values on top of bars
for i, v in enumerate(drivers_to_watch['PositionDelta']):
    plt.text(i, v + 0.1, f"+{v:.1f}", color='white', ha='center')
    
# Add grid and predicted positions
for i, (grid, pred) in enumerate(zip(drivers_to_watch['GridPosition'], drivers_to_watch['PredictedPosition'])):
    plt.text(i, -0.5, f"Grid: P{grid:.0f} → Pred: P{pred:.1f}", color='white', ha='center')

# Customize the plot
plt.ylabel('Predicted Positions Gained', fontsize=12)
plt.title('Drivers to Watch - 2025 Saudi Arabian Grand Prix', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.2)
plt.box(False)

# Add Otto Rentals caption
plt.figtext(0.5, 0.01, 
           "Powered by https://www.otto.rentals – Best and safest way to rent a car in Kenya", 
           ha='center', 
           fontsize=10, 
           color='#2E5EAA',
           weight='bold')

plt.tight_layout(pad=2.0)
plt.subplots_adjust(bottom=0.1)  # Make room for the caption

# Save the figure
plt.savefig('saudi_gp_drivers_to_watch_2025.png', dpi=300, bbox_inches='tight')
plt.show()


# ===== VISUALIZATION 5: PODIUM PROBABILITY =====
# Calculate a simple "podium probability" based on predicted position
# The closer to position 1, the higher the probability
top_drivers = prediction_df.head(8).copy()
max_pos = top_drivers['PredictedPosition'].max()
top_drivers['PodiumProbability'] = ((max_pos - top_drivers['PredictedPosition'] + 1) / max_pos * 100).round(1)

plt.figure(figsize=(12, 6))
plt.style.use('dark_background')

# Create the probability bars
bars = plt.bar(top_drivers['DriverName'], 
              top_drivers['PodiumProbability'], 
              color=top_drivers['Color'], 
              alpha=0.8)

# Add percentage labels
for i, v in enumerate(top_drivers['PodiumProbability']):
    plt.text(i, v + 1, f"{v}%", color='white', ha='center')

# Customize the plot
plt.ylabel('Podium Probability (%)', fontsize=12)
plt.title('Podium Contenders - 2025 Saudi Arabian Grand Prix', fontsize=14)
plt.grid(axis='y', linestyle='--', alpha=0.2)
plt.ylim(0, 105)  # Set y-axis limit to accommodate percentage labels

# Add Otto Rentals caption
plt.figtext(0.5, 0.01, 
           "Powered by https://www.otto.rentals – Best and safest way to rent a car in Kenya", 
           ha='center', 
           fontsize=10, 
           color='#2E5EAA',
           weight='bold')

plt.tight_layout(pad=2.0)
plt.subplots_adjust(bottom=0.1)  # Make room for the caption

# Save the figure
plt.savefig('saudi_gp_podium_probability_2025.png', dpi=300, bbox_inches='tight')
plt.show()


# ===== VISUALIZATION 6: TRACK MAP WITH DRIVER POSITIONS =====
# Create a simple visual representation of the Jeddah circuit with predicted top positions
plt.figure(figsize=(12, 8))
plt.style.use('dark_background')

# Helper function to create a simplified track layout
def create_track_path():
    theta = np.linspace(0, 2*np.pi, 100)
    
    # Create basic oval
    r = 1
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    # Add some corners to make it look more like Jeddah
    x = x * 1.5
    y = y * 1
    
    # Add a chicane
    chicane_idx = 20
    x[chicane_idx:chicane_idx+10] += 0.2 * np.sin(np.linspace(0, np.pi, 10))
    
    # Add another corner
    corner_idx = 60
    x[corner_idx:corner_idx+15] -= 0.3 * np.sin(np.linspace(0, np.pi, 15))
    
    return x, y

# Create the track
track_x, track_y = create_track_path()
plt.plot(track_x, track_y, 'w-', linewidth=15, alpha=0.3)
plt.plot(track_x, track_y, 'w-', linewidth=13, alpha=0.5)

# Add start/finish line
start_x, start_y = track_x[0], track_y[0]
plt.plot([start_x-0.1, start_x+0.1], [start_y-0.05, start_y+0.05], 'w-', linewidth=2)
plt.text(start_x, start_y+0.2, "START/FINISH", color='white', ha='center', fontsize=10)

# Place top 5 drivers on track
top_drivers = prediction_df.head(5)
positions = np.linspace(0, 80, len(top_drivers))

for i, (driver, team, color) in enumerate(zip(top_drivers['DriverName'], top_drivers['Team'], top_drivers['Color'])):
    # Place markers at different positions along the track
    marker_idx = int(positions[i])
    if marker_idx >= len(track_x):
        marker_idx = marker_idx % len(track_x)
    
    plt.plot(track_x[marker_idx], track_y[marker_idx], 'o', markersize=12, color=color)
    
    # Add driver name next to marker (shorten to last name only)
    last_name = driver.split(' ')[-1]
    offset_x = 0.1 * np.cos(2*np.pi*marker_idx/len(track_x))
    offset_y = 0.1 * np.sin(2*np.pi*marker_idx/len(track_x))
    plt.text(track_x[marker_idx] + offset_x, track_y[marker_idx] + offset_y, 
             f"P{i+1} {last_name}", color='white', ha='center', fontsize=10)

# Add race info
plt.text(0, -1.5, "JEDDAH CORNICHE CIRCUIT", color='white', ha='center', fontsize=16, weight='bold')
plt.text(0, -1.7, "2025 SAUDI ARABIAN GRAND PRIX", color='white', ha='center', fontsize=14)

# Remove axes
plt.axis('off')
plt.axis('equal')  # Equal aspect ratio

# Add Otto Rentals caption
plt.figtext(0.5, 0.01, 
           "Powered by https://www.otto.rentals – Best and safest way to rent a car in Kenya", 
           ha='center', 
           fontsize=10, 
           color='#2E5EAA',
           weight='bold')

plt.tight_layout(pad=2.0)
plt.subplots_adjust(bottom=0.1)  # Make room for the caption

# Save the figure
plt.savefig('saudi_gp_track_prediction_2025.png', dpi=300, bbox_inches='tight')
plt.show()