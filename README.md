# AI-Cyber-Defense-Firewall

Objective
To detect "Zero-Day" anomalies in real-time using an unsupervised Autoencoder model.

Features
Unsupervised Learning: Trained on benign traffic to establish a behavioral baseline.

Real-Time Sniffing: Uses Scapy and Npcap for live packet interception.

Tiered Sensitivity: Implements context-aware thresholds (0.02 for Admin, 0.08 for Web) to reduce false positives.

Results
The system successfully identified 11,199 threats in simulated attack data and provides millisecond-level verdicts on live traffic.
