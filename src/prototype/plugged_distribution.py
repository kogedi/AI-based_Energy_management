#!/usr/bin/python3

import csv
import numpy as np
from datetime import datetime


class plugged_distribution:
    def calc_plug_distribution(self,id):
        # 15min intervalls for the whole day
        I = 24 * 4
        # The plugged distribution for each device_id
        plugged_distributions = {}

        with open('data/first_tier_data_set.csv') as f:
            reader = csv.reader(f, delimiter=',')
            for row in reader:
                if row[0] == 'device_id':
                    continue

                device_id = row[0]
                start_time = datetime.fromisoformat(row[1][:-4])
                end_time = datetime.fromisoformat(row[2][:-4])

                if device_id not in plugged_distributions:
                    plugged_distributions[device_id] = np.zeros((I))

                t_start = start_time.hour * 60 + start_time.minute
                t_end = end_time.hour * 60 + end_time.minute
                for i in range(I):
                    t = 15 * i
                    if t_start <= t and t <= t_end:
                        plugged_distributions[device_id][i] += 1 

        for key, val in plugged_distributions.items():
            s = 0
            for i in range(I):
                s += val[i]
            for i in range(I):
                p = val[i] * 100 / s
                plugged_distributions[key][i] = (p >= 1.0)               
        for i in range(I):
            v = plugged_distributions[id][i]
            
        
        return plugged_distributions[id][:]
        
def main():
    return 0           
if __name__ == "__main__":
    main()