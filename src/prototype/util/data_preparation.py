def data_preparation(df):
    df = df.drop(['min_charge_mW', 'max_charge_mW', 'max_charge_mW', 'max_discharge_mW', 'charging_efficiency', 'discharging_efficiency'], axis=1)
    df['time_difference'] = (df['end_time']-df['start_time']).dt.total_seconds()
    df = df.drop(['start_time', 'end_time'], axis = 1)
    return df
