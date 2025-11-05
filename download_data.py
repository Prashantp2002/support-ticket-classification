# download_data.py — offline sample dataset
import os
import pandas as pd

rows = [
    ["VPN keeps disconnecting after login", "Network"],
    ["Forgot password; need reset", "Access"],
    ["Laptop keyboard not working", "Hardware"],
    ["Outlook not syncing emails", "Software"],
    ["Need license for Photoshop", "Software"],
    ["Invoice shows wrong amount", "Billing"],
    ["Cannot connect to office Wi-Fi", "Network"],
    ["Request access to Git repository", "Access"],
    ["Laptop overheating frequently", "Hardware"],
    ["Payment charged twice on card", "Billing"],
    ["Teams call drops every 5 minutes", "Network"],
    ["Account locked after MFA change", "Access"],
]
os.makedirs("data", exist_ok=True)
pd.DataFrame(rows, columns=["text", "category"]).to_csv("data/tickets.csv", index=False)
print("Sample dataset created at data/tickets.csv")
