import pandas as pd
import numpy as np
import datetime

class scTier():
    def timesToPluggedDis(self, arrival_time, departure_time, device_today, data_set, date):
        df = pd.read_csv(data_set, sep=',', header=0)
        data_set_start = df["start_time"]
        data_set_end = df["end_time"]
        device_id = np.array(df["device_id"])

        I = 24*4

        date = datetime.date(date[0], date[1], date[2])
        print(date)

        end_date = np.array(pd.to_datetime(data_set_end).dt.date)
        print(end_date)

        data_set_end = np.array(pd.to_datetime(data_set_end))

        today = []

        for i in range(len(end_date)):
            if (end_date[i] == date):
                today.append(i)
        data_set_start = np.array(pd.to_datetime(data_set_start))

        start_time = []
        end_time = []
        device_id_today = []

        for i in range(len(today)):
            start_time.append(data_set_start[today[i]])
            end_time.append(data_set_end[today[i]])
            device_id_today.append(device_id[today[i]])

        print(start_time)
        print(end_time)
        print(device_id_today)


        

        results_D1 = []


        for i in range(3):
            if i+1 in device_id_today:
                for j in range(len(device_id_today)):
                    if device_id_today[j] == i+1:
                        index = j
                    
                this_device = []
                for k in range(I):
                    t = 15*k
                    # t_start = -((t*I) - start_time[index].hour * 60 + start_time[index].minute)
                    t_start = 0
                    print(t_start)
                    
                    t_end = end_time[index].hour * 60 + end_time[index].minute
                    print(t_end)
                    if t_start <= t and t <= t_end:
                        print("smaller")
                        this_device.append(1)
                    else:
                        this_device.append(0)
                results_D1.append(np.array(this_device))
                        
            else:
                results_D1.append(np.zeros((I)))

        print(results_D1)

        


def main():
    tier = scTier()
    tier.timesToPluggedDis(arrival_time=0, departure_time=0, data_set="data/first_tier_data_set.csv", date=[2024, 3, 5], device_today=0)
    return 0           
if __name__ == "__main__":
    main()