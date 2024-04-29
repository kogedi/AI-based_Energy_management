#!/usr/bin/python3

import csv
import numpy as np
from datetime import datetime

# 15min intervalls for the whole day
I = 24 * 4
# The plugged distribution for each device_id

def extract_cost_vector(filename):
    with open(filename) as f:
        reader = csv.reader(f, delimiter=',')
        vec = []
        for row in reader:
            if row[0] == 'datetime_utc':
                continue

            t = datetime.fromisoformat(row[0][:-4])
            if not (t.month == 3 and t.day == 5):
                continue
            #print(t)

            cost = float(row[1])
            vec.append(cost)
            vec.append(cost)
            vec.append(cost)
            vec.append(cost)

        return vec

# print('prices actual:')
# print(extract_cost_vector('data/first_tier_prices_actual.csv'))
# print()
# print()
# print()
# print('prices forecast:')
# print(extract_cost_vector('data/first_tier_prices_forecast.csv'))
