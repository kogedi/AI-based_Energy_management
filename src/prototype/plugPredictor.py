import numpy as np
import pandas as pd
# import os #maybe later for authentification
# from google.cloud import bigquery
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

def bigquery():
    # Create a BigQuery client object.
    client = bigquery.Client()

    # Perform a query.
    QUERY = (
        'SELECT * FROM `chefreff-hack24ham-3805.1k5_data.first_tier_data_set` '
    )
    query_job = client.query(QUERY)  # API request
    rows = query_job.result()  # Waits for query to finish

    # Convert the query results to a Pandas DataFrame.
    df = rows.to_dataframe()
    return df

# Read CSV file
def read_csv(filename):
    try:
        df = pd.read_csv(filename, sep=',', header=0)
        return df
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return None
    
def modify_dates(dates, start_end):
    dates = pd.to_datetime(dates)
    years = dates.dt.year - 2000
    months = dates.dt.month
    days = dates.dt.day
    weekdays = dates.dt.weekday
    hours = dates.dt.hour
    minutes = dates.dt.minute
    dict = {f"year_{start_end}": years, f"month_{start_end}": months, f"day_{start_end}": days, f"weekday_{start_end}": weekdays, f"hour_{start_end}": hours, f"minute_{start_end}": minutes}
    return pd.DataFrame(dict)

def normalize(input_data, norm):
    n = (1 / max(input_data)) * norm
    return n

class NeuralNet(nn.Module):
    def __init__(self, nbr_input_colmn):
        super(NeuralNet, self).__init__()

        # First layer of the network
        # nbr_input_column inputs, 20 outputs
        self.fc1 = nn.Linear(nbr_input_colmn, 40) 
        
        # Second layer of the network
        # 20 inputs, 20 outputs
        self.fc2 = nn.Linear(40,60)
        
        # 20 inputs, 20 outputs
        self.fc3 = nn.Linear(60,202)
        
        # 20 inputs, 1 output
        self.fc4 = nn.Linear(202,1)
        
        
        # Relu activation function
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(p=0.29)

    
    def forward(self, x):
        
        # Apply the network to a given matrix of inputs x
        
        out = self.fc1(x) # apply first layer
        out = self.relu(out) # apply activation function
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = self.relu(out)
        out = self.fc4(out)
        return out

    
data_total = read_csv("data/first_tier_data_set.csv")

norm = 512


start_time = modify_dates(data_total["start_time"], "start")
end_time = modify_dates(data_total["end_time"], "end")
kWh_charged = data_total["kWh_charged"]

data_total['time_difference'] = (pd.to_datetime(data_total['end_time']) - pd.to_datetime(data_total['start_time'])).dt.total_seconds()
data_total["time_difference"] = normalize(data_total["time_difference"], norm)

print("Header of data:",
    list(data_total.columns.values))

nn_data_header = pd.concat([start_time, end_time], axis=1)
nn_data_header = pd.concat([nn_data_header, data_total["device_id"]], axis=1)
nn_data_header = pd.concat([nn_data_header, data_total['time_difference']], axis=1)
# nn_data_header = pd.concat([nn_data_header, data_total['min_charge_mW'] * (1/max(data_total["min_charge_mW"])) * 60], axis=1)
# nn_data_header = pd.concat([nn_data_header, data_total['max_charge_mW'] * (1/max(data_total["max_charge_mW"])) * 60], axis=1)
nn_data_header = pd.concat([nn_data_header, kWh_charged], axis=1).drop(["year_start", "minute_start", "minute_end", "year_end", "day_start", "day_end", "month_start", "month_end", "hour_end", "hour_start", "weekday_start", "weekday_end"], axis=1)
print(nn_data_header)

print("Header of data:",
    list(data_total.columns.values))

nn_data = nn_data_header.sample(frac=1).reset_index(drop=True)
nn_data = np.array(nn_data)
print(nn_data)

print("data",
    nn_data.shape)

n_len = nn_data.shape
train_fraction = 0.8
train_amount = int(n_len[0] * train_fraction)

input_columns = n_len[1]-1


# take the first 3000 examples for training
X_train = nn_data[:train_amount,: input_columns] # all features except last column
y_train = nn_data[:train_amount, input_columns]  # kWh column

# and the remaining examples for testing
X_test = nn_data[train_amount:,:input_columns] # all features except last column
y_test = nn_data[train_amount:,input_columns] # kWh column


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
print("device=",device)

# Create network object
model = NeuralNet(input_columns).to(device)

# Adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)


# Loss function
criterion = nn.MSELoss()

# Training with batches

# how many epochs to train
n_epochs = 60

batch_size = 15

train_examples = X_train.shape[0]

n_batches = int(train_examples/batch_size)

# Keep track of the losses 
train_losses = []
test_losses = []

# Already move the full tensors to GPU to simplify things later on
X_train_cuda = torch.tensor(X_train,dtype=torch.float)#.cuda()
y_train_cuda = torch.tensor(y_train,dtype=torch.float)#.cuda()

# Loop over the epochs
for ep in range(n_epochs):
                
    # Each epoch is a complete loop over the training data
    for i in range(n_batches):
        
        # Reset gradient
        optimizer.zero_grad()
        
        i_start = i*batch_size
        i_stop  = (i+1)*batch_size
        
        # Convert x and y to proper objects for PyTorch
        x = X_train_cuda[i_start:i_stop]
        y = y_train_cuda[i_start:i_stop]

        # Apply the network 
        net_out = model(x)
                
        # Calculate the loss function
        loss = criterion(net_out,y)
                
        # Calculate the gradients
        loss.backward()
        
        # Update the weights
        optimizer.step()
        
        
    # Calculate predictions for the full training and testing sample
    # y_pred_train = model(torch.tensor(X_train,dtype=torch.float).cuda()).cpu().detach().numpy().flatten()
    # y_pred = model(torch.tensor(X_test,dtype=torch.float).cuda()).cpu().detach().numpy().flatten()

    y_pred_train = model(torch.tensor(X_train,dtype=torch.float)).cpu().detach().numpy().flatten()
    y_pred = model(torch.tensor(X_test,dtype=torch.float)).cpu().detach().numpy().flatten()

    # Calculate aver loss / example over the epoch
    train_loss = np.mean((y_pred_train-y_train)**2)
    test_loss = np.mean((y_pred-y_test)**2)
    
    # print some information
    print("Epoch:",ep, "Train Loss:", train_loss,  "Test Loss:", test_loss)
    
    # and store the losses for later use
    train_losses.append(train_loss)
    test_losses.append(test_loss)
    

# After the training:
    
# Prepare scatter plot
# y_pred =  model(torch.tensor(X_test,dtype=torch.float).cuda()).cpu().detach().numpy().flatten()

y_pred = model(torch.tensor(X_test,dtype=torch.float)).cpu().detach().numpy().flatten()



print("Best loss:", min(test_losses), "Final loss:", test_losses[-1])

print("Correlation coefficient:", np.corrcoef(y_pred,y_test)[0,1])
plt.scatter(y_pred_train,y_train)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()

# Prepare and loss over time
plt.plot(train_losses,label="train")
plt.plot(test_losses,label="test")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()


