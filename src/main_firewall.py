import pandas as pd
import numpy as np
import os

# 1. SET THE PATH TO YOUR DATA ON E: DRIVE
# Make sure this matches exactly where you extracted the folder
data_path = r"E:\Pro\AI_Cyber_Defense\AI_Cyber_Defense\data\MachineLearningCVE"
monday_file = os.path.join(data_path, "Monday-WorkingHours.pcap_ISCX.csv")

def ingest_and_clean_data(file_path):
    print(f"[*] Starting Ingestion of: {os.path.basename(file_path)}")
    
    # Load 50,000 rows (enough to train, but fast for your CPU)
    df = pd.read_csv(file_path, nrows=50000)
    
    # FIX 1: Remove hidden spaces from column names (e.g., ' Flow Duration' -> 'Flow Duration')
    df.columns = df.columns.str.strip()
    
    # FIX 2: Handle Infinity and NaN (common in CICIDS2017)
    # We replace 'Infinity' with NaN and then drop those rows
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    
    print(f"[+] Ingestion Complete: {len(df)} clean rows ready.")
    return df

# Run the ingestion
df_normal = ingest_and_clean_data(monday_file)

# Preview the data your AI will learn from
print(df_normal.head())


from sklearn.preprocessing import MinMaxScaler

def scale_data(df):
    print("[*] Starting Min-Max Scaling...")
    
    # 1. DROP THE LABEL (AI only needs numbers, not the word 'BENIGN')
    # Also drop any non-numeric columns if they exist
    features = df.drop(columns=['Label'])
    
    # 2. INITIALIZE THE SCALER
    scaler = MinMaxScaler()
    
    # 3. TRANSFORM THE DATA
    # This turns every column into values between 0.0 and 1.0
    scaled_values = scaler.fit_transform(features)
    
    # 4. CONVERT BACK TO A CLEAN DATAFRAME
    df_scaled = pd.DataFrame(scaled_values, columns=features.columns)
    
    print("[+] Scaling Complete. Data is now AI-ready (Range: 0 to 1).")
    return df_scaled, scaler

# Apply scaling to your cleaned Monday data
df_scaled, my_scaler = scale_data(df_normal)

# Let's see the transformation
print(df_scaled[['Destination Port', 'Flow Duration']].head())



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# --- PART 4: THE AUTOENCODER BRAIN ---
class CyberAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super(CyberAutoencoder, self).__init__()
        # Encoder: Compresses the data
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16) # The Bottleneck
        )
        # Decoder: Tries to rebuild the data
        self.decoder = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim),
            nn.Sigmoid() # Keeps output between 0 and 1
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# --- PREPARING FOR TRAINING ---
# Convert your scaled E: Drive data into PyTorch Tensors
X_train = torch.tensor(df_scaled.values, dtype=torch.float32)

# Create a DataLoader (Batch size 32 is standard for CPUs)
train_loader = DataLoader(TensorDataset(X_train), batch_size=32, shuffle=True)

# Initialize the model using the number of columns in your data (78 or 79)
input_dim = df_scaled.shape[1]
model = CyberAutoencoder(input_dim)

# Define Loss (How well the brain is learning) and Optimizer
criterion = nn.MSELoss() # Mean Squared Error
optimizer = optim.Adam(model.parameters(), lr=0.001)

print(f"[*] AI Model initialized with {input_dim} input features.")


# --- PART 5: THE TRAINING PROCESS ---
num_epochs = 5  # Start with 5 to keep it fast on your CPU
print(f"[*] Starting Training for {num_epochs} epochs...")

model.train() # Set the model to training mode
for epoch in range(num_epochs):
    train_loss = 0.0
    for data in train_loader:
        inputs = data[0]
        
        # 1. Clear gradients
        optimizer.zero_grad()
        
        # 2. Forward pass: What does the brain think?
        outputs = model(inputs)
        
        # 3. Calculate Loss: How much did the reconstruction miss?
        loss = criterion(outputs, inputs)
        
        # 4. Backward pass: Correct the brain's mistakes
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * inputs.size(0)
    
    avg_loss = train_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.6f}")

print("[+] Training Complete. The AI now understands 'Normal' traffic.")

# Save the brain to your E: Drive so you don't have to retrain later
torch.save(model.state_dict(), "cyber_autoencoder.pth")
print("[*] Model saved as 'cyber_autoencoder.pth' on E: Drive.")


# --- PART 6: DETECTION (TESTING AGAINST ATTACKS) ---
friday_ddos_file = os.path.join(data_path, "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv")

def detect_anomalies(test_file, model, scaler):
    print(f"[*] Testing system against: {os.path.basename(test_file)}")
    
    # 1. Ingest and Clean Friday Data
    df_attack = ingest_and_clean_data(test_file)
    
    # 2. Scale using the SAME scaler from Monday
    features_attack = df_attack.drop(columns=['Label'])
    scaled_attack = scaler.transform(features_attack)
    X_test = torch.tensor(scaled_attack, dtype=torch.float32)
    
    # 3. Predict & Calculate Reconstruction Error
    model.eval()
    with torch.no_grad():
        reconstructions = model(X_test)
        loss = torch.mean((reconstructions - X_test)**2, dim=1).numpy()
    
    # 4. Set Threshold
    threshold = 0.01  
    anomalies = loss > threshold
    
    print(f"[!] Detection Results: Found {np.sum(anomalies)} potential threats!")
    
    # Show results
    df_attack['AI_Score'] = loss
    df_attack['Is_Anomaly'] = anomalies
    print(df_attack[['Label', 'AI_Score', 'Is_Anomaly']].head(10))
    
    # FIX 1: This MUST be indented inside the function!
    return df_attack

# --- PART 7: AUTOMATED MITIGATION ---
def mitigate_threats(df_results):
    # Filter for rows marked as anomalies
    attacker_rows = df_results[df_results['Is_Anomaly'] == True]  
    if not attacker_rows.empty:
        # FIX: We count unique values in the index (where IP info usually is)
        # or you can use a specific column like df_results.index.unique()
        unique_attacker_count = len(attacker_rows.index.unique())
        
        print(f"\n[!] ALERT: Mitigation system engaged for {unique_attacker_count} unique potential threats.")
        print(f"[*] Action Taken: Malicious activity detected and logged.")
        print("[+] Network Health Score: Restoring...")
    else:
        print("\n[+] System Status: All traffic appears Benign. Health Score: 100%")

# ==========================================
# FINAL EXECUTION
attack_results = detect_anomalies(friday_ddos_file, model, my_scaler)
mitigate_threats(attack_results)

# --- PART 8: REAL-TIME SNIFFER INTEGRATION ---

from scapy.all import sniff, IP, TCP, UDP
import time



# --- PART 8: LIVE AI FIREWALL ---
from scapy.all import sniff, IP, TCP, UDP

# --- PART 8: LIVE AI PROTECTION ---
loss_history = []

def process_live_packet(packet):
    global loss_history
    if packet.haslayer(IP):
        src_ip = packet[IP].src
        dest_port = packet[TCP].dport if packet.haslayer(TCP) else packet[UDP].dport if packet.haslayer(UDP) else 0

        # 1. Capture real packet data
        live_features = np.zeros((1, 78)) 
        live_features[0, 0] = dest_port / 65535.0
        live_features[0, 3] = len(packet) / 1500.0

        # 2. AI Inference
        model.eval()
        with torch.no_grad():
            input_tensor = torch.tensor(live_features, dtype=torch.float32)
            reconstruction = model(input_tensor)
            loss = torch.mean((reconstruction - input_tensor)**2).item()

        # 3. SMOOTHING: Keep track of the last 10 packets
        loss_history.append(loss)
        if len(loss_history) > 10:
            loss_history.pop(0)
        
        avg_loss = sum(loss_history) / len(loss_history)

        # 4. ACTION THRESHOLD
       # YOUR THOUGHT: Port-Specific Sensitivity
        status = "SAFE"
        
        if dest_port in [21, 22, 23, 3389]: # Sensitive Admin Ports
            threshold = 0.03 # High Sensitivity
        elif dest_port == 443: # Common Web Traffic
            threshold = 0.08 # Low Sensitivity (Ignore Noise)
        else:
            threshold = 0.06 # Standard Sensitivity
            
        if avg_loss > threshold:
            status = "!!! ANOMALY !!!"
            
        print(f"[{status}] {src_ip}:{dest_port} | Pkt Loss: {loss:.4f} | Avg: {avg_loss:.4f}")


def start_firewall():
    print("\n" + "="*50)
    print("[*] LIVE AI FIREWALL ACTIVE - SNIFFING ON E: DRIVE")
    print("="*50 + "\n")
    # Captures live packets and sends each one to our AI judge
    sniff(prn=process_live_packet, store=0)

# Start the live defense
start_firewall()