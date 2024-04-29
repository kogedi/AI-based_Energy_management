#!/usr/bin/python3

"""Setup and implementation of an optimization
"""
import numpy
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.optimize import Bounds
from plugged_distribution import *
from  util import extract_prices
from battery_distribution import *
from max_power_distribution import *
from datetime import datetime
from util import data_preparation

class first_tier:
    def scheduleEnergyConsumption(self, pct, pft, plugged_D_1, plugged_D, kWh_charged, min_charge_Power, max_charge_Power, eta_c, eta_dc,deviceID,T, full_printer,full_plot):
        """
        #************ IMPLEMENTATION Outline ******************
        #
        # 0 - setup variables
        # 1 - objective
        # 2 - start x0
        # 3 - bounds
        # 4 - constraints
        # 5 - MINIMIZE OPTIMIZATION
        # 6 - resulting vector
        # 7 - plot x
        #******************************************************

        Parameters
        ----------
        pct : array
            cleared price of day D-1
        pft : array
            forecasted price of day D
        plugged_D_1 : array, boolean values of [0,1]
            plugged: 1, unplugged: 0
        plugged_D : array, boolean values of [0,1]
            plugged: 1, unplugged: 0
        kWh_charged: 
            predicted energy charged within one charging time
        min_charge_Power: float in kW
            minimal charge power allowed while charging
        max_charge_Power: float in kW
            maximal charge power while charging
        eta_c: float in [0,1]
            efficiency coefficient of charging
        eta_dc: float in [0,1]
            efficiency coefficient of discharging
        deviceID: integer
            ID of the vehicle to schedule energy consumption
        T: float
            time step for scheduling eg. 0.25 corresponds to 15 min
        full_printer: boolean
            print every scheduled and feed in value
        full_plot: boolean
            plot every plot
            
        Returns
        -------
        list
            return a list of a scheduled energy consumption of the next day
        """

        # 0 - setup variables
        N = len(4*pct) # Dimension of the problem
        I = int(round(24/T))
        alpha = numpy.ones(N)
        #alpha vector = [Esc(t)d-1, Eds(t)d-1, Esc(t)d, Eds(t)d].T
        pct_plus = [x + 0.14 for x in pct]
        pct_minus = [x for x in pct]
        pft_plus = [x + 0.14 for x in pft]
        pft_minus = [x for x in pft]
        p = pct_plus + pct_minus + pft_plus + pft_minus
        # plt.plot(p)
        # plt.show()
        plugged_D_1_list = plugged_D_1.tolist()
        plugged_D_list = plugged_D.tolist()
        #penalty = 10000
        plugged = plugged_D_1_list + plugged_D_1_list + plugged_D_list + plugged_D_list
        # plt.plot(plugged)
        # plt.show()
        #ones to zero and zeros  to ones
        plugged_inv = np.zeros(len(plugged))
        for i in range(len(plugged)):
            if plugged[i] == 0:
                plugged_inv[i] = 1 # penalty
            elif plugged[i] >0:
                plugged_inv[i] = 0
        #1 - objective
        objective = lambda alpha: numpy.dot(alpha.T,p) 
        
        # 2 - start
        start = numpy.ones(N) #Why this start vector step-up. Maybe better np.zeros(N)

        # 3 - bounds
        #Slack variable defined as a global variable of SVM
        n = len(alpha) // 4
        #B = [(min_charge_Power, max_charge_Power) for b in range(n)]
        B = [(0, max_charge_Power) for b in range(n)]
        B = B + [(0,max_charge_Power) for b in range(n)]
        #B = B + [(min_charge_Power, max_charge_Power) for b in range(n)]
        B = B + [(0,max_charge_Power) for b in range(n)]
        B = B + [(0,max_charge_Power) for b in range(n)]

        # 4 - constraints

        zerofun = lambda alpha : numpy.sum(T*eta_c*alpha[:n]) - numpy.sum(T*eta_dc*alpha[n:2*n]) \
        + numpy.sum(T*eta_c*alpha[2*n:3*n]) -  numpy.sum(T*eta_dc*alpha[3*n:]) - kWh_charged
        
        fun = lambda alpha : numpy.dot(alpha.T, plugged_inv)
        #constraints
        XC = ({'type':'eq', 'fun':zerofun}, {'type':'eq', 'fun':fun}) #contraints zerofun to be ZERO


        # 5 - MINIMIZE (1, 2, 3, 4)
        ret = minimize(objective, start, bounds=B, constraints = XC)
        
        # 6 - resulting vector for minimal cost
        alphamin = ret['x']
        minimal_value = objective(ret['x'])
        planned_energy = numpy.sum(T*alphamin[:n]) - numpy.sum(T*alphamin[n:2*n]) + numpy.sum(T*alphamin[2*n:3*n]) - numpy.sum(T*alphamin[3*n:])
        print("Planned energy", "%.2f" % planned_energy)
        print("Minimum of the optimization function:","%.2f" % minimal_value,"\n")
        
        # 7 - plot schedule for the next day
        E_c_schedule_D_1 = alphamin[:n]
        E_c_schedule_D = alphamin[2*n:3*n]
        E_dc_schedule_D_1 = alphamin[n:2*n]
        E_dc_schedule_D = alphamin[3*n:]
        
        if full_printer:
            print("power_supply_scheduled_kW", E_c_schedule_D)
            print("power_feed_in_scheduled_kW", E_dc_schedule_D,"\n")

        E_c_plot = E_c_schedule_D 
        E_dc_plot = E_dc_schedule_D 
        p_plot = p[n:2*n]
        plugged_plot = plugged[n:2*n]
        
        two_day_plot = False # TODO if two day plot with day D-1 and D should be plotted
        if two_day_plot:
            E_c_plot = numpy.concatenate((E_c_schedule_D_1, E_c_schedule_D))
            E_dc_plot = numpy.concatenate((E_dc_schedule_D_1, E_dc_schedule_D))
            p_plot = numpy.concatenate(extract_prices.extract_cost_vector('data/first_tier_prices_actual.csv'),extract_prices.extract_cost_vector('data/first_tier_prices_forecast.csv'))
            p_plot = numpy.concatenate((p[:n],p[n:2*n]))
            plugged_plot = numpy.concatenate((plugged[:n], plugged[n:2*n]))
        
        # txt output
        schedule_output_txt = False
        if schedule_output_txt:
            file_path = "output.txt"
            # Save the array to a text file
            np.savetxt(file_path, E_c_schedule_D, fmt='%d')  # '%d' specifies integer format

        # plot Result
        if full_plot:
            plt.plot(E_c_plot, label='E_charge', color='blue')
            plt.plot(E_dc_plot, label='E_discharge', color='green')
            plt.plot(p_plot, label='Price', color='red')
            plt.plot(plugged_plot, label='plugged', color='grey')
            plt.legend()
            plt.title(f'Scheduled Energy Demand for Every timestep | Vehicle {deviceID}')
            plt.xlabel('Time t')
            plt.ylabel('Energy E_t')
            plt.savefig(f"figures/figure{deviceID}")
            plt.show()
        
        # create RESULTS output for every T minutes (15 min in this case)
        hours_per_intervall = T # 15min intervalls
        results = []
        # initial value guess
        energy_stored_expected_kWh = max(0, (numpy.sum(E_c_schedule_D_1) - numpy.sum(E_dc_schedule_D_1)) * hours_per_intervall)
        for i in range(I):
            total_minutes = i * 15
            hours = total_minutes // 60
            minutes = total_minutes % 60
            t1 = datetime(year=2024, month=3, day=6, hour=hours, minute=minutes, second=0)
            time_start_utc = t1.strftime("%Y-%m-%d %H:%M:%S UTC")

            total_minutes = (i+1) * 15
            day=6
            hours = total_minutes // 60
            minutes = total_minutes % 60
            if hours >= 24:
                day=7
                hours=0
                minutes=0
            t2 = datetime(year=2024, month=3, day=day, hour=hours, minute=minutes, second=0)
            time_end_utc = t2.strftime("%Y-%m-%d %H:%M:%S UTC")

            energy_stored_expected_kWh += eta_c * E_c_schedule_D[i] * hours_per_intervall
            energy_stored_expected_kWh -= eta_dc * E_dc_schedule_D[i] * hours_per_intervall

            item = {
                'team_name': 'The Ants',
                'device_id': deviceID,
                'time_start_utc': time_start_utc,
                'time_end_utc': time_end_utc,
                'plug_in_state': plugged[i],
                'power_supply_scheduled_kW': E_c_schedule_D[i],
                'power_feed_in_scheduled_kW': E_dc_schedule_D[i],
                'energy_stored_expected_kWh': energy_stored_expected_kWh,
            }
            results.append(item)

        return results

def main():
    
    choose_standard_parameters = True
    if choose_standard_parameters:
        eta_c = 0.92
        eta_dc = 0.88
        T = 0.25 # time step granularity
        full_printer = False # TODO: False to accelerate the program
        full_plot = False # TODO: False to accelerate the program
        
    # energy prices for today and tomorrow
    pct = extract_prices.extract_cost_vector('data/first_tier_prices_actual.csv')
    pft = extract_prices.extract_cost_vector('data/first_tier_prices_forecast.csv')
    
    ft = first_tier()
    
    results = []
    
    print("\nOPTIMIZATION of scheduled charging energy (time+amount) for different vehicles\n")
    print("1. Set full_plot = True/False, full_printer = True/False for more/less details")
    print("2. Change choose_standard_parameters if necessary")
    print("3. Change two_day_plot = True to have a look into the first day")
    print("4. Use EXAMPLE_Values = False for debugging\n")
    for i in range(1,8): # TODO hardcoded for 7 devices

        deviceID = str(i)
        
        # Calculate expected plugged in array for Day D and Day D-1
        pd = plugged_distribution()
        plugged_D_1 = pd.calc_plug_distribution(deviceID)
        plugged_D = pd.calc_plug_distribution(deviceID)
        
        EXAMPLE_Values = False # TODO for fast debugging 
        if EXAMPLE_Values:
            pct = [0.2, 0.3, 0.4, 0.4]
            pft = [0.3, 0.4, 0.5, 0.6]
            plugged_D_1 = [0, 0, 1, 1]
            plugged_D = [0, 0, 1, 1]
            
        kWh_charged = battery_distribution()[deviceID] #*1000000
        
        min_charge_Power = 2 # TODO Standard value
        max_charge_Power = max_power_distribution()[deviceID]/1000000

        print("Vehicle",deviceID)
        print('Estimated demand', "%.2f" % kWh_charged)

        # Create an instance of first tier class
        
        results += ft.scheduleEnergyConsumption(pct, pft, plugged_D_1, plugged_D, kWh_charged, min_charge_Power, \
                                                max_charge_Power,eta_c, eta_dc,deviceID, T, full_printer,full_plot)

    return results        

if __name__ == "__main__":
    
    results = main()
    
    
    ## JSON Write *********************************************
    WRITE_JSON = True # TODO depending on the desired output 

    if WRITE_JSON:
        import json
        # write results to json file for tier one
        
        with open('results/first_tier_expected_values.json', 'w') as f:
            json.dump(results, f)
    
    
    ## CSV Write **********************************************
    WRITE_CSV = True # TODO depending on the desired output
    
    if WRITE_CSV: 
        import csv
        # Assuming `results` is a list of dictionaries where each dictionary represents a row in the CSV
        # Specify the field names (column names) for your CSV
        field_names = results[0].keys()  # Assuming all dictionaries in `results` have the same keys

        # Write the results to a CSV file
        with open('results/first_tier_expected_values.csv', 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=field_names)
            # Write the header row
            writer.writeheader()
            
            # Write the data rows
            for row in results:
                writer.writerow(row)