# -*- coding: utf-8 -*-
"""
Created on Wed Jul  8 14:46:36 2026

@author: shame
"""

import pandas as pd
import os
data_path = "data/1_bronze/train_FD001.txt" if os.path.exists("data/1_bronze/train_FD001.txt") else "train_FD001.txt"
raw_data = pd.read_csv(data_path, sep=" ", header=None)

clean_column = []

for i in range(26):
    clean_column.append(i)
    

data_table = raw_data[clean_column]

column_names = ["engine_id","cycle_number","setting_1","setting_2","setting_3"]
for sensor_number in range(1,22):
    column_names.append("sensor_"+ str(sensor_number))


data_table.columns = column_names

all_engine_list = []

for engine_num in range(1,101):
    single_engine = data_table[data_table["engine_id"] == engine_num]
    
    feature_only = single_engine.loc[:,"sensor_1":"sensor_21"]
    
    engine_matrix = feature_only.to_numpy()
    
    all_engine_list.append(engine_matrix)


total_engine = len(all_engine_list)

print("Total Number of engines: ", total_engine)

short_run = 999999
long_run = 0

for matrix in all_engine_list:
    number_of_rows = matrix.shape[0]
    
    if number_of_rows < short_run:
        short_run = number_of_rows
        
    if number_of_rows > long_run:
        long_run = number_of_rows
        
print("Shortest enigne sequence: ", short_run)
print("Longest engine Sequence: ", long_run)

engine_ids = []
max_cycles = []

for num in range(1,101):
    single_engine = data_table[data_table["engine_id"] == num]
    
    lifetime = single_engine["cycle_number"].max()
    
    engine_ids.append(f"Engine {num}")
    max_cycles.append(lifetime)
    
for engine_id in range(1,101):
    each_engine = max_cycles[engine_id-1]
    print(f"No of cycles for engine {engine_id} : {each_engine}")
    






