#!/usr/bin/python3

import csv
import numpy as np
from datetime import datetime

def battery_distribution():
    battery_distributions = {}
    with open('data/first_tier_data_set.csv') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            device_id = row[0]
            if device_id == 'device_id':
                continue
            
            battery = float(row[6])
                
            if device_id not in battery_distributions:
                battery_distributions[device_id] = [0, 0.0]
            
            battery_distributions[device_id][0] += 1
            battery_distributions[device_id][1] += battery

        results = {}
        for key in battery_distributions.keys():
            results[key] = battery_distributions[key][1] / battery_distributions[key][0]

    return results
