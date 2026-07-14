import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

raw_data = pd.read_csv("train_FD001.txt", sep=" ", header=None)

clean_columns = []

for i in range(26):
    clean_columns.append(i)
    
    
data_table = raw_data[clean_columns]

column_names = ["engine_id", "cycle_number", "setting_1", "setting_2", "setting_3"]

for sensor_number in range(1,22):
    column_names.append("sensor_"+str(sensor_number))
    
data_table.columns=column_names

MAX_RUL = 130

all_engine_features = []
all_engine_targets = []

for engine_num in range(1,101):
    single_engine = data_table[data_table["engine_id"] == engine_num]
    
    
    feature_only = single_engine.loc[:,"setting_1":"sensor_21"]
    engine_matrix = feature_only.to_numpy()
    
    
    t_eol = single_engine["cycle_number"].max()
    cycle_numbers = single_engine["cycle_number"].to_numpy()
    linear_rul = t_eol - cycle_numbers
    
    piecewise_rul = np.minimum(linear_rul, MAX_RUL)
    
    all_engine_features.append(engine_matrix)
    all_engine_targets.append(piecewise_rul)
    
    
    
print("Total Engines Processed: ", len(all_engine_targets))
print("MAX_RUL ceiling threshold: ", MAX_RUL)

all_end_at_zero = all(target[-1] == 0 for target in all_engine_targets)
print("All engines end at RUL = 0: ", all_end_at_zero)

global_max = max(target.max() for target in all_engine_targets)
print("Global maximum RUL observed: ", global_max)



engine_1 = data_table[data_table["engine_id"]==1]
cycles = engine_1["cycle_number"].to_numpy()
sensor_11 = engine_1["sensor_11"].to_numpy()


t_oel_1 = cycles.max()
linear_rul_1 = t_oel_1 - cycles
piecewise_rul_1 = np.minimum(linear_rul_1,MAX_RUL)

inflection_cycle = t_oel_1-MAX_RUL

fig, (ax1, ax2) = plt.subplot(2,1, figsize=(10,8), sharex=True)

ax1.plot(cycles, linear_rul_1, label="Linear RUL (Unrealistic Baseline)", color="gray", linestyle="--", linewidth=1.5)
ax1.plot(cycles, piecewise_rul_1, label="Heimes Piecewise RUL (MAX_RUL=130)", color="#1f77b4", linewidth=2.5)
ax1.axvline(x=inflection_cycle, color="red", linestyle=":", label=f"Inflection Point (Cycle {inflection_cycle})")
ax1.set_ylabel("Remaining Useful Life (Cycles)", fontsize=11, fontweight="bold")
ax1.set_title("Engine #1: Target Label Engineering vs. Physical Degradation", fontsize=13, fontweight="bold")
ax1.legend(loc="upper right")
ax1.grid(True, linestyle="--", alpha=0.5)

ax2.plot(cycles, sensor_11, color="#ff7f0e", linewidth=2.0, label="Sensor 11 (Static Pressure)")
ax2.axvline(x=inflection_cycle, color="red", linestyle=":", label=f"Inflection Point (Cycle {inflection_cycle})")
ax2.set_xlabel("Engine Flight Cycle Number", fontsize=11, fontweight="bold")
ax2.set_ylabel("Sensor 11 Pressure (psia)", fontsize=11, fontweight="bold")
ax2.legend(loc="upper left")
ax2.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()
