import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

df = pd.read_csv('Public_Transport.csv')

df['Metropolitan train'] = df['Metropolitan train'].str.replace(',', '').fillna('0').astype(int)
df['Metropolitan tram'] = df['Metropolitan tram'].str.replace(',', '').fillna('0').astype(int)
df['Metropolitan bus'] = df['Metropolitan bus'].str.replace(',', '').fillna('0').astype(int)
df['Regional train'] = df['Regional train'].str.replace(',', '').fillna('0').astype(int)
df['Regional coach'] = df['Regional coach'].str.replace(',', '').fillna('0').astype(int)
df['Regional bus'] = df['Regional bus'].str.replace(',', '').fillna('0').astype(int)

# Create a new column for total passengers by summing all transport-related columns
df['Total passengers'] = (df['Metropolitan train'] + df['Metropolitan tram'] +
                          df['Metropolitan bus'] + df['Regional train'] +
                          df['Regional coach'] + df['Regional bus'])

# Limit the data 
df = df[df['_id'] <= 78]

# Plotting the trend 
plt.figure(figsize=(10,6))
plt.plot(df['_id'], df['Total passengers'], marker='o', linestyle='-', color='b')
plt.title('Trend of Total Passengers Using Public Transport')
plt.xlabel('Year')
plt.ylabel('Total Number of Passengers (in Millions)')

plt.gca().yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, _: f'{x*1e-6:.0f}M'))
plt.xticks(df['_id'], df['Year'], rotation=45, ha='right')
plt.grid(True)
plt.tight_layout()
plt.show()