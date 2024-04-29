"""Setup and implementation of an optimization
"""
import numpy
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.optimize import Bounds
from plugged_distribution import *
from prototype.util.extract_prices import *
from battery_distribution import *
from max_power_distribution import *

class first_tier:
    def scheduleEnergyConsumption(self, pct, pft, plugged_D_1, plugged_D, kWh_charged, min_charge_Power, max_charge_Power, eta_c, eta_dc,deviceID,T,E_c_schedule_D):
        """
        #************ IMPLEMENTATION Outline ******************
        #
        # 0 - setup variables
        # 1 - objective
        # 2 - start x0
        # 3 - bounds
        # 4 - constraints
        # 5 - MINIMIZE function (1, 2, 3, 4)
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
        """
                
        
        # 0 - setup variables
        N = len(pct) # Dimension of the problem
        alpha = numpy.ones(N)
        #alpha vector = [Esc(t)d-1, Eds(t)d-1, Esc(t)d, Eds(t)d].T
        pct_plus = [x + 0.14 for x in pct]
        pct_minus = [x for x in pct]
        # pft_plus = [x + 0.14 for x in pft]
        # pft_minus = [x for x in pft]
        p = pct_plus + pct_minus #+ pft_plus + pft_minus
        # plt.plot(p)
        # plt.show()
        plugged_D_1_list = plugged_D_1.tolist()
        plugged_D_list = plugged_D.tolist()
        #penalty = 10000
        plugged = plugged_D_1_list + plugged_D_1_list #+ plugged_D_list + plugged_D_list
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
        objective = lambda alpha: numpy.dot(E_c_schedule_D,p) + numpy.linalg.norm(E_c_schedule_D,alpha)*0.8
        
        # 2 - start
        start = numpy.ones(N) #Why this start vector step-up. Maybe better np.zeros(N)

        # 3 - bounds
        #Slack variable defined as a global variable of SVM
        #n = len(alpha) // 4
        #B = [(min_charge_Power, max_charge_Power) for b in range(n)]
        B = [(0, max_charge_Power) for b in range(N)]
        # B = B + [(0,max_charge_Power) for b in range(n)]

        # #B = B + [(min_charge_Power, max_charge_Power) for b in range(n)]
        # B = B + [(0,max_charge_Power) for b in range(n)]
        # B = B + [(0,max_charge_Power) for b in range(n)]

        # 4 - constraints

        # zerofun = lambda alpha : numpy.sum(T*eta_c*alpha[:n]) - numpy.sum(T*eta_dc*alpha[n:2*n]) \
        # + numpy.sum(T*eta_c*alpha[2*n:3*n]) -  numpy.sum(T*eta_dc*alpha[3*n:]) - kWh_charged
        
        fun = lambda alpha : numpy.sum(alpha)*T - kWh_charged
        #constraints
        XC = ( {'type':'eq', 'fun':fun}) 
        # {'type':'eq', 'fun':zerofun},
        # contraints zerofun to be ZERO


        # 5 - MINIMIZE (1, 2, 3, 4)
        ret = minimize(objective, start, bounds=B, constraints = XC)#,method='trust-constr')
        #ret2 = minimize(objective, start, bounds=B, constraints = XC)
        # 6 - resulting vector
        alphamin = ret['x']
        minimal_value = objective(ret['x'])
        #planned_energy = numpy.sum(T*alphamin[:n]) - numpy.sum(T*alphamin[n:2*n]) + numpy.sum(T*alphamin[2*n:3*n]) - numpy.sum(T*alphamin[3*n:])
        energy_realized = numpy.sum(alphamin)*T
        print("Energy demanded", kWh_charged)
        print("Energy charged",energy_realized )
        print("minimal_value",minimal_value)
        # 7 - plot alphamin

        #plt.plot(alphamin, label='E_charge', color='blue')
        # E_c_schedule_D_1 = alphamin[:n]
        # E_c_schedule_D = alphamin[2*n:3*n]
        # E_dc_schedule_D_1 = alphamin[n:2*n]
        # E_dc_schedule_D = alphamin[3*n:]
        # print("power_supply_scheduled_kW",E_c_schedule_D)
        # print("power_feed_in_scheduled_kW",E_dc_schedule_D)
        
        #TODO T
        # T_15 = 0,25
        # for
        #     energy_stored_expected_kWh = numpy.sum(E_c_schedule_D_1*T_15) - numpy.sum(E_dc_schedule_D_1*T_15) + eta_c * E_c_schedule_D*T_15 - eta_dc * E_dc_schedule_D*T_15
        
        E_c_plot = E_c_schedule_D #numpy.concatenate((E_c_schedule_D_1, E_c_schedule_D))
        E_dc_plot = E_dc_schedule_D  #numpy.concatenate((E_dc_schedule_D_1, E_dc_schedule_D))
        #price_plot = numpy.concatenate(extract_cost_vector('data/first_tier_prices_actual.csv'),extract_cost_vector('data/first_tier_prices_forecast.csv')))
        p_plot = p[n:2*n] #numpy.concatenate((p[:n],p[n:2*n]))
        plugged_plot = plugged[n:2*n] #numpy.concatenate((plugged[:n], plugged[n:2*n]))
        
        plt.plot(E_c_plot, label='E_charge', color='blue')
        plt.plot(E_dc_plot, label='E_discharge', color='green')
        plt.plot(p_plot, label='Price', color='red')
        plt.plot(plugged_plot, label='plugged', color='grey')
        plt.legend()
        # Add title and labels
        plt.title('Scheduled Energy Demand for Every timestep | Vehicle')#, deviceID)
        plt.xlabel('Time t')
        plt.ylabel('Energy E_t')
        plt.show()
        #plt.savefig(deviceID) # TODO

def main():
    for i in range(1,8):
        
        deviceID = str(i)
        print("deviceID",deviceID)
        # Example arrays
        #pct = [0.2, 0.3, 0.4, 0.4]
        pct = extract_cost_vector('data/first_tier_prices_actual.csv')
        #pft = [0.3, 0.4, 0.5, 0.6]
        pft = extract_cost_vector('data/first_tier_prices_forecast.csv')
        #plugged_D_1 = [0, 0, 1, 1]
        #plugged_D = [0, 0, 1, 1]
        pd = plugged_distribution()
        plugged_D_1 = pd.calc_plug_distribution(deviceID)
        plugged_D = pd.calc_plug_distribution(deviceID)
        kWh_charged = battery_distribution()[deviceID] #*1000000
        print('My estimated demand', kWh_charged)
        min_charge_Power = 2
        max_charge_Power = max_power_distribution()[deviceID]/1000000
        eta_c = 0.92
        eta_dc = 0.88
        T = 0.25
        E_c_schedule_D = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 3.0204739481033183e-12, 3.0336853430289882e-12, 3.0282804334202434e-12, 3.0310742230628306e-12, 1.0591662195850895e-11, 1.0567230270335017e-11, 1.0590908085721337e-11, 1.0582419011455944e-11, \
            1.033541011850228e-11, 1.0345705692536139e-11, 1.0335525561445372e-11, 6.776284538151825e-11, 7.714304671029394e-11, 7.715173887924935e-11, 7.71276592832964e-11, 7.716012592027507e-11, 1.0265672171634538e-10, 1.0266760262927103e-10, 1.0265094601804523e-10, 1.0262753406277976e-10, 1.4419356137831204e-10, 1.441938465505688e-10, 1.4419726407869088e-10, 1.4421233019038722e-10, 10.697231268375745, \
            10.697231268375766, 10.697231268375752, 10.697231268375692, 10.697231268300925, 10.697231268300829, 10.69723126830091, 10.69723126830093, 0.12660710073750758, 0.12663968528227731, 0.12661624044382003, 0.12663968528233993, 1.144831611099297e-10, 1.1449251490858509e-10, 1.1449907599853712e-10, \
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        # Create an instance of firsttier class
        ft = first_tier()

        # Call scheduleEnergyConsumption function
        ft.scheduleEnergyConsumption(pct, pft, plugged_D_1, plugged_D, kWh_charged, min_charge_Power, max_charge_Power,eta_c, eta_dc,deviceID,T,E_c_schedule_D)
        
        

if __name__ == "__main__":
    main()