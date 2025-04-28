import re
import json
import csv

# Path to your log file
log_file_path = "output.txt"  # <-- change this to your actual file

# Pattern to capture metrics for each epoch
pattern = re.compile(
    r"epoch (\d+), mean training loss: ([\d.]+).*?"
    r"mean testing loss: ([\d.]+), average running time: ([\d.]+)s.*?"
    r"MSE: ([\d.]+).*?"
    r"MaskedMSE: ([\d.]+).*?"
    r"RMSE: ([\d.]+).*?"
    r"MaskedRMSE: ([\d.]+).*?"
    r"REL: ([\d.]+).*?"
    r"MaskedREL: ([\d.]+).*?"
    r"MAE: ([\d.]+).*?"
    r"MaskedMAE: ([\d.]+).*?"
    r"Threshold@1.05: ([\d.]+).*?"
    r"MaskedThreshold@1.05: ([\d.]+).*?"
    r"Threshold@1.10: ([\d.]+).*?"
    r"MaskedThreshold@1.10: ([\d.]+).*?"
    r"Threshold@1.25: ([\d.]+).*?"
    r"MaskedThreshold@1.25: ([\d.]+)",
    re.DOTALL
)

# Read file
with open(log_file_path, "r", encoding="utf-8", errors="ignore") as f:
    content = f.read()

# Remove ANSI escape codes
ansi_escape = re.compile(r'\x1B\[[0-?]*[ -/]*[@-~]')
cleaned_content = ansi_escape.sub('', content)

# Extract matches
results = []
for match in pattern.finditer(cleaned_content):
    results.append({
        "Epoch": int(match.group(1)),
        "Train Loss": float(match.group(2)),
        "Test Loss": float(match.group(3)),
        "Test Time": float(match.group(4)),
        "MSE": float(match.group(5)),
        "Masked MSE": float(match.group(6)),
        "RMSE": float(match.group(7)),
        "Masked RMSE": float(match.group(8)),
        "REL": float(match.group(9)),
        "Masked REL": float(match.group(10)),
        "MAE": float(match.group(11)),
        "Masked MAE": float(match.group(12)),
        "Threshold@1.05": float(match.group(13)),
        "Masked Threshold@1.05": float(match.group(14)),
        "Threshold@1.10": float(match.group(15)),
        "Masked Threshold@1.10": float(match.group(16)),
        "Threshold@1.25": float(match.group(17)),
        "Masked Threshold@1.25": float(match.group(18)),
    })

# Save as CSV
with open("parsed_metrics_1.csv", "w", newline='') as csvfile:
    fieldnames = results[0].keys()
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for row in results:
        writer.writerow(row)

# Optionally also save as JSON
with open("parsed_metrics_1.json", "w") as jsonfile:
    json.dump(results, jsonfile, indent=2)

print(f"Extracted {len(results)} epochs. Saved as parsed_metrics.csv and parsed_metrics.json")
