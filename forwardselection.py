import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load dataset
df = pd.read_csv("D:/Laptop backup/Winter 2025/MSc DS 2/Programs/auto-mpg.csv")

# Replace '?' with NaN
df.replace('?', np.nan, inplace=True)

# Convert horsepower to numeric
df['horsepower'] = pd.to_numeric(df['horsepower'])

# Drop missing values
df = df.dropna()

# Drop non-numeric column (car name)
if 'car name' in df.columns:
    df = df.drop(columns=['car name'])

# Define features and target
X = df.drop(columns=['mpg'])
y = df['mpg']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#remaining_features → features not yet selected
#selected_features → features already selected
#best_score → best R² score achieved so far
#We initialize best_score = -∞ so that the first feature always improves the score.

remaining_features = list(X.columns)
selected_features = []
best_score = -np.inf

print("Forward Feature Selection Process:\n")


#The loop continues until:
#No features remain, OR
#Adding a new feature does not improve performance

while remaining_features:
    scores = []

   #We temporarily add one feature at a time to the selected set. 
    for feature in remaining_features:
        features_to_test = selected_features + [feature]
        
        #Train regression model using only the selected + candidate feature.
        #This is the wrapper mechanism (model-based evaluation).
        model = LinearRegression()
        model.fit(X_train[features_to_test], y_train)
        
        y_pred = model.predict(X_test[features_to_test])
        score = r2_score(y_test, y_pred)
        
        scores.append((score, feature))
    
    #Sort features by R² score.
    #Select feature giving highest improvement.
    scores.sort(reverse=True)
    current_best_score, best_feature = scores[0]


#If adding the feature improves R²:
#Update best_score
#Add feature permanently
#Remove from remaining list
#Else:
#Stop algorithm (no further improvement)

    
    if current_best_score > best_score:
        best_score = current_best_score
        selected_features.append(best_feature)
        remaining_features.remove(best_feature)
        
        print(f"Added: {best_feature}, R2 Score: {best_score:.4f}")
    else:
        break

print("\nFinal Selected Features:")
print(selected_features)
