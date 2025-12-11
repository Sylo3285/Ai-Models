import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def generate_synthetic_data():
    n = 20000
    df = pd.DataFrame(columns=['cost','fuel','toll','time','labor'])

    for i in tqdm(range(n)):
        
        # --- 1. Trip distance in km (base variable) ---
        distance = np.random.normal(loc=120, scale=40)   # avg 120km trips
        distance = max(distance, 10)                     # no negative values

        # --- 2. Fuel usage (liters) ---
        fuel_efficiency = np.random.normal(5, 0.8)       # liters per km
        fuel = distance * fuel_efficiency                
        fuel += np.random.normal(0, 5)                   # noise
        fuel = max(fuel, 20)

        # --- 3. Time (minutes) ---
        avg_speed = np.random.normal(60, 10)             # km/h realistic speed
        time = (distance / avg_speed) * 60               # min
        time += np.random.normal(0, 10)
        time = max(time, 15)

        # --- 4. Toll cost ---
        # Probability of toll increases with distance
        base_toll = np.random.choice([0, 1], p=[0.6, 0.4])
        toll = base_toll * np.random.uniform(30, 250)

        # --- 5. Labor cost ---
        labor_rate = np.random.uniform(200, 500)         # per hour
        labor = labor_rate * (time / 60)
        labor += np.random.normal(0, 20)
        labor = max(labor, 50)

        # --- 6. Final cost formula ---
        cost = (fuel * 1.2) + toll + labor               # fuel price ~1.2x
        
        df.loc[i] = [cost, fuel, toll, time, labor]

    df.to_csv('data.csv', index=False)
    print("Synthetic data generated and saved to data.csv")

def analyze_data():
    df = pd.read_csv('data.csv')
    print("Data Summary:")
    print(df.describe())
    # --- SCATTER SPREAD PLOT ---
    plt.figure(figsize=(12,6))

    for col in df.columns:
        plt.scatter(df.index, df[col], s=10, label=col)

    plt.xlabel("Index")
    plt.ylabel("Value")
    plt.title("Scatter Spread of Cost Components")
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    generate_synthetic_data()
    analyze_data()