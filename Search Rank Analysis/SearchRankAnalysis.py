import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_excel('Search Rank Analysis/Trend Dataset.xlsx')

# Display the first few rows of the dataframe to understand its structure
# Make sure to replace this with appropriate checks or exploration of the data.
print(df.head())
print(df.info())

# Remove rows where 'Division' is 'others'
df_new = df[df['Division'] != 'others']

# Select relevant columns and drop 'Department'
df_rel = df_new[['Search Term', 'Division', 'Search Frequency Rank', 'Department', 'Category']]
df_rel = df_rel.drop('Department', axis=1)

# Create a 'Relevant_Freq_Rank' column based on the row order
df_rel['Relevant_Freq_Rank'] = range(1, df_rel.shape[0] + 1)

# Calculate 'Trend_Score' based on the 'Relevant_Freq_Rank'
df_rel['Trend_Score'] = df_rel.shape[0] / df_rel['Relevant_Freq_Rank']

# Rank within each 'Division' based on 'Relevant_Freq_Rank' using dense ranking
df_rel['Division_Freq_Rank'] = df_rel.groupby('Division')['Relevant_Freq_Rank'].rank("dense")

# Rank within each 'Category' based on 'Relevant_Freq_Rank' using dense ranking
df_rel['Category_Freq_Rank'] = df_rel.groupby('Category')['Relevant_Freq_Rank'].rank("dense")

# Calculate the maximum rank for each 'Division'
Max_Rank_For_Each = df_rel.groupby('Division')['Division_Freq_Rank'].max()

# Print the results for verification
print("Max Rank For Each Division:")
print(Max_Rank_For_Each)

# Now let's visualize the trend scores and the rankings

# Bar plot to visualize average 'Trend_Score' per 'Division'
plt.figure(figsize=(12, 6))
sns.barplot(data=df_rel, x='Division', y='Trend_Score', estimator=np.mean)
plt.title('Average Trend Score per Division')
plt.xlabel('Division')
plt.ylabel('Average Trend Score')
plt.xticks(rotation=45)
plt.show()

# Bar plot to visualize the frequency rank distribution per Division
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_rel, x='Division', y='Trend_Score')
plt.title('Trend Score Distribution per Division')
plt.xlabel('Division')
plt.ylabel('Trend Score')
plt.xticks(rotation=45)
plt.show()

# Scatter plot to visualize the relationship between 'Relevant_Freq_Rank' and 'Trend_Score'
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_rel, x='Relevant_Freq_Rank', y='Trend_Score', hue='Division', palette='viridis')
plt.title('Scatter Plot of Relevant Freq Rank vs Trend Score')
plt.xlabel('Relevant Frequency Rank')
plt.ylabel('Trend Score')
plt.legend(title='Division')
plt.show()

# Scatter plot to visualize the relationship between 'Category_Freq_Rank' and 'Trend_Score'
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_rel, x='Category_Freq_Rank', y='Trend_Score', hue='Category', palette='coolwarm')
plt.title('Scatter Plot of Category Freq Rank vs Trend Score')
plt.xlabel('Category Frequency Rank')
plt.ylabel('Trend Score')
plt.legend(title='Category')
plt.show()
