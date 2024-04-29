import os

from flask import Flask, request, jsonify
from google.cloud import bigquery
import torch
from datetime import datetime
import numpy as np

import first_tier

app = Flask(__name__)

I = 24 * 4

example_data_first = [
    {
        'team_name': 'The Ants',
        'device_id': '0',
        'time_start_utc': '2024-03-06 00:00:00 UTC',
        'time_end_utc': '2024-03-06 00:15:00 UTC',
        'plug_in_state': False,
        'power_supply_scheduled_kW': 0.0,
        'power_feed_in_scheduled_kW': 0.0,
        'energy_stored_expected_kWh': 0.0,
    }
]

example_data_second = [
    {
        'team_name': 'The Ants',
        'device_id': '0',
        'time_start_utc': '2024-03-06 00:00:00 UTC',
        'time_end_utc': '2024-03-06 00:15:00 UTC',
        'power_supply_realized_kW': 0.0,
        'power_feed_in_realized_kW': 0.0,
        'energy_stored_realized_kWh': 0.0,
    }
]

def calc_plugged(data):
    plugged_distributions = {}
    for row in data:
        device_id = row.device_id
        if device_id == 'device_id':
            continue
        
        start_time = row.start_time
        end_time = row.end_time

        if device_id not in plugged_distributions:
            plugged_distributions[device_id] = np.zeros((I)).tolist()

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

    return plugged_distributions

@app.route("/handin/first")
def api_handin_first():
    return first_tier.main()

@app.route("/handin/second")
def api_handin_second():
    return example_data_second

@app.route('/learn')
def api_learn():
    client = bigquery.Client()
    QUERY = (
        'SELECT * FROM `chefreff-hack24ham-3805.1k5_data.first_tier_data_set` '
    )
    query_job = client.query(QUERY)
    data = query_job.result()

    plugged = calc_plugged(data)
    return plugged

@app.route('/test/pytorch')
def api_test_pytorch():
    x = torch.rand(5, 3)
    n = torch.norm(x).numpy()[0]
    return {'torch': n}

@app.route('/test/bigquery')
def api_test_bigquery():
    client = bigquery.Client()

    # Perform a query.
    QUERY = (
        'SELECT * FROM `chefreff-hack24ham-3805.1k5_data.first_tier_data_set` '
    )
    query_job = client.query(QUERY)  # API request
    rows = query_job.result()  # Waits for query to finish

    result = []
    for row in rows:
        result.append({
            'start_time': row.start_time
        })

    return result


@app.route("/")
def hello_world():
    return f"<h1>The Ant Server</h1>"

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
