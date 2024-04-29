#!/usr/bin/python3

import csv
import numpy as np
from datetime import datetime

def max_power_distribution():
    dists = {}
    with open('data/first_tier_data_set.csv') as f:
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            device_id = row[0]
            if device_id == 'device_id':
                continue
            
            max_power = float(row[4])
                
            if device_id not in dists:
                dists[device_id] = [0, 0.0]
            
            dists[device_id][0] += 1
            dists[device_id][1] += max_power

        results = {}
        for key in dists.keys():
            results[key] = dists[key][1] / dists[key][0]

        return results