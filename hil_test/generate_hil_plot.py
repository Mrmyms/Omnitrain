import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('master_hil_summary.csv', header=None, names=['Model', 'SuccessRate', 'Fitness', 'Distance'])
df['Model'] = df['Model'].str.replace('.omnibit', '')

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

bars = ax.bar(df['Model'], df['Distance'], color=['#B0B0B0', '#B0B0B0', '#2E8B57', '#FF9999', '#FF9999', '#FF9999'], edgecolor='black', width=0.5, linewidth=1.2)

for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 5, f"{yval:.1f}m", ha='center', va='bottom', fontweight='bold')

ax.set_ylabel('Mean Distance (m)')
ax.set_title('F1TENTH HIL Performance (ESP32-S3 Zero-Allocation)')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
plt.xticks(rotation=25)
plt.tight_layout()
plt.savefig('/Users/mr.myms/Omnitrain/paper_exports/hil_performance_updated.png', bbox_inches='tight')
print("Plot saved.")
