import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import tree
import joblib
import warnings

warnings.filterwarnings("ignore")

# Load and display data
music_data = pd.read_csv(
    "https://raw.githubusercontent.com/mosh-hamedani/python-supplementary-materials/main/music.csv"
)
print(f"Music data:\n{music_data.values}")

# Prepare data and train model
X = music_data.drop(columns=["genre"])
y = music_data["genre"]

model = DecisionTreeClassifier()
model.fit(X, y)

input_data = [[21, 1], [22, 0]]
predictions = model.predict(input_data)
print(f"Input data: {input_data}")
print(f"Predictions: {predictions}")

# Split data and evaluate model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model.fit(X_train, y_train)
predictions = model.predict(X_test)
score = accuracy_score(y_test, predictions)
print(f"Score: {score}")

# Save model
joblib.dump(model, "music-recommender.joblib")

# Load model and make prediction
model = joblib.load("music-recommender.joblib")
input_data = [[21, 1]]
predictions = model.predict(input_data)
print(f"Predictions after loading model:")
print(f"    Input data: {input_data}")
print(f"    Predictions: {predictions}")

# Export decision tree visualization
tree.export_graphviz(
    model,
    out_file="music-recommender.dot",
    feature_names=["age", "gender"],
    class_names=sorted(y.unique()),
    label="all",
    rounded=True,
    filled=True,
)
