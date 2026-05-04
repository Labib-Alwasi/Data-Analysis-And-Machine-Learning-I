# Section 1: Data Analysis

## Dataset: HBS03.csv

**Source:** [Irish Central Statistics Office (CSO)](https://www.cso.ie/en/)  
**Survey:** Household Budget Survey 2023

### Description

Average household expenditure across 2023 divided into quintiles. Households are ranked by income spending — the 1st quintile spends the least, the 5th quintile represents the highest earners. Expenditure data is structured hierarchically, where each category breaks down into subcategories.

> Logically, lower quintiles prioritise food, housing, fuel, light, and transport. Remaining income determines miscellaneous and household goods spending, which increases significantly for higher quintiles.

---

## Real-World Problems Addressed

### 1. Cost of Living Analysis
Identifies cost of living by quintile and year. Useful for energy companies optimising electricity supply, and data analysts modelling spending habits by region or identifying households with the most disposable income.

### 2. Expenditure Analysis for Businesses
Helps businesses identify popular products at specific times of year. Airlines and hotels can predict peak travel periods; supermarkets can stock seasonal products; businesses can tailor financial strategies accordingly.

### 3. Social Welfare Programmes
Enables governments to identify lower-income households spending a disproportionate share of income on essentials (food, energy). Supports prediction of household financial stress adjusted for inflation and cost-of-living changes, allowing targeted welfare intervention.

---

## Pre-processing

**Group by quintile:**
```python
df_year = grouped_by_quintile.get_group('All quintiles').copy()
```

**Extract top-level hierarchy** (rows with no sub-categories):
```python
df_top_level = df_year[df_year['Expenditure'].apply(lambda x:
    len(x.split('.')) == 1)].copy()
```

**Shorten verbose category names and rename columns:**
```python
df['Expenditure'] = df['Expenditure'].replace({
    'Total miscellaneous goods, services and other expenditure (09)': 'Miscellaneous (09)'
})
df = df.rename(columns={'Household Disposable Income Quintiles': 'Quintiles'})
```

**Pivot for cross-quintile comparison:**
```python
df_pivot = df_comp.pivot(index='Quintiles', columns='Expenditure', values='VALUE')
```

---

## Visualisations

### Heatmap
![Heatmap](https://github.com/user-attachments/assets/0b733f8a-b0c5-4c6a-a0cc-aca3b4e83f73)

### Bar Charts
![Bar Chart 1](https://github.com/user-attachments/assets/9a2335e9-07d2-4656-92ee-895b8a7b689f)
![Bar Chart 2](https://github.com/user-attachments/assets/352f5468-b738-414f-ad0a-8af3ead7c254)

### Geographical Visualisation

Expenditure values are binned into ranges for choropleth mapping:

```python
bins   = [0, 50, 100, 150, 200, 300, 500, 700]
labels = ['0–50', '50–100', '100–150', '150–200', '200–300', '300–500', '500–700']
map_merged['VALUE_BIN'] = pd.cut(map_merged['VALUE'], bins=bins, labels=labels, include_lowest=True)
```

![Map](https://github.com/user-attachments/assets/69256627-12e2-42df-b2c0-3b947c4c5b55)

# Section 2: Machine Learning on Sensor Dataset
Traffic data across 22 months. No traffic across first few months likely due to road closed off. 3
sensors were placed across 3 different road. They point towards the pedestrian, inbound and
outbound roads. The presence of vehicles in the pedestrian road suggests the sensor was
entirely focused on this particular road, other environmental factors that moved the sensor, or
that vehicles were moving on or dangerously close to the pedestrian’s pathway. The countline_id
indicates each sensor. However each sensor points towards a specific road, which is much
more useful data to work with.

##  Formulate two different real-world problems
Classification. We can classify the level of traffic congestion, eg. Low, medium and high
congestion. This can also help what sort of vehicles pass through these roads at what time of
the day. This can also help up learn when heavy goods vehicles use these roads on a particular
day. Another useful benefit is if traffic congestion regularly surpasses what the road was
designed for. In this scenario, city observers could alert the authorities and demand necessary
regulations to mitigate the probability of road accidents occurring.
We can extrapolate and predict traffic trends for these particular roads. We can forecast when
traffic congestion through these roads will be, and plan accordingly. This would be useful data
that navigation providers would want to help keep drivers up to date with the latest traffic. Civil
engineers could use this information to insert the necessary traffic control to reduce road
accidents and divert traffic to the most appropriate directions.

## Pre-processing 

### Feature Engineering

New time-based columns are added to support aggregation and visualization:

```python
# Add monthly, daily, and hourly columns
df['month'] = df['time_from'].dt.to_period('M')
df['daily'] = df['time_from'].dt.to_period('D')
df['hour'] = df['time_from'].dt.to_period('h')
```

These features enable easier grouping for analysis and plotting.

---

### Data Aggregation

The original dataset records traffic every 5 minutes, which is too granular for broader insights. The data is aggregated into monthly, daily, and hourly summaries.

```python
# Columns to sum
cols_to_sum = [
    'car', 'bus', 'cyclist', 'motorbike',
    'pedestrian', 'rigid', 'total_count'
]

# Monthly aggregation
df_monthly = df.groupby(
    [df['time_from'].dt.to_period('M'), 'name', 'countline_id']
)[cols_to_sum].sum().reset_index()

# Daily aggregation
df_daily = df.groupby(
    [df['time_from'].dt.to_period('D'), 'name', 'countline_id']
)[cols_to_sum].sum().reset_index()

# Hourly aggregation
df_hourly = df.groupby(
    [df['time_from'].dt.to_period('H'), 'name', 'countline_id']
)[cols_to_sum].sum().reset_index()
```

---

### Data Cleaning

There are no missing values in the dataset. However, the first 9 months contain only zero traffic counts, which are not useful.

After inspection, traffic activity begins on **October 11th at 00:00**, indicating when the road became active.

```python
# Remove initial zero-traffic period
df_monthly = df_monthly[df_monthly['time_from'] >= '2024-09']
df_daily = df_daily[df_daily['time_from'] >= '2024-10-11']
df_hourly = df_hourly[df_hourly['time_from'] >= '2024-10-11']
```

---

### Traffic Classification

A categorical `traffic_level` feature is added to the daily dataset. Traffic levels are defined using quantiles:

```python
df_daily = df_daily.copy()

df_daily['traffic_level'] = pd.qcut(
    df_daily['total_count'],
    q=3,
    labels=['Low', 'Medium', 'High']
)
```

---

### Feature Scaling & Model Pipeline

To prevent bias toward more frequent vehicle types (e.g., cars), numerical features are scaled before training.

A machine learning pipeline is constructed using scaling and multinomial logistic regression:

```python
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(multi_class="multinomial", max_iter=100))
])
```

Scaling ensures all features contribute equally to the model, improving fairness and performance.

## Data Visualisation

<img width="628" height="479" alt="image" src="https://github.com/user-attachments/assets/809e2239-ae00-4830-a3d1-62aaefb73762" />
<img width="611" height="513" alt="image" src="https://github.com/user-attachments/assets/d29d6b20-3843-4f4e-a441-cb5d1fa28309" />
<img width="738" height="528" alt="image" src="https://github.com/user-attachments/assets/9ab26550-9c09-4ee6-8955-ea13497f69ee" />
<img width="687" height="433" alt="image" src="https://github.com/user-attachments/assets/dfc014ed-4907-4842-9e0c-0f3166011217" />
<img width="617" height="524" alt="image" src="https://github.com/user-attachments/assets/78a1f11f-8649-4aea-826e-bb27270b105c" />

## Training a Machine Learning Model

We train a **Random Forest Regressor** to predict pedestrian volume based on traffic features.

```python
# Use only existing numeric columns that work with Random Forest
feature_columns = ['day', 'car', 'bus', 'cyclist', 'motorbike', 'rigid']

X = df[feature_columns]
y = df['pedestrian']  # Target: pedestrian volume

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Make predictions
y_pred = rf_model.predict(X_test)

# Evaluation metrics
print(f"R² Score: {r2_score(y_test, y_pred):.4f}")

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"RMSE: {rmse:.2f}")
print(f"MSE: {mse:.2f}")

# Feature importance
prediction = pd.DataFrame({
    'Vehicle': feature_columns,
    'Occurrence': rf_model.feature_importances_
}).sort_values('Occurrence', ascending=False)

print("\nPredicted Occurrence of each vehicle")
print(prediction)
```

### Results

- **R² Score:** 0.8367  
- **RMSE:** 103.60  
- **MSE:** 10732.92  

### Predicted Occurrence of Each Vehicle

| Rank | Vehicle    | Occurrence |
|------|-----------|------------|
| 1    | cyclist   | 0.384457   |
| 2    | car       | 0.348819   |
| 3    | motorbike | 0.115237   |
| 4    | bus       | 0.102322   |
| 5    | day       | 0.031702   |
| 6    | rigid     | 0.017462   |



# Section 3 Machine Learning on Geographical Data
The goal is to gather satellite images and apply clustering and prediction algorithms. Images of
urban settlements surrounded by vegetation is gathered. They are images of Navan shopping
centre. No ground truths are provided.

3.2. Formulate two different real-world problems (Clustering - Unsupervised and Classification-Supervised)

Supervised learning can be used to classify pixels based on ground truths. It does so when a
training set is provided and create a model that can generalise data on unseen dataset. For
satellite images, data can be obtained from multi-spectral bands which provide more precise
data along with information such as urban settlements that companies such as Google or NASA
provide but can’t be easily obtained from screenshots.
Unsupervised Machine Learning is useful and preferred when there is no ground truths labelled
in datasets. In raw satellite image that contain urban, vegetation and water, this is often the
case. Therefore, clustering is introduced. This is an algorithm that attempts to find similar pixels
patterns that could represent the images, making it possible to discriminate different regions. K-
Means is one such algorithm where similar pixels gather around centroids. These centroids are
initially random, but they are algorithmically computed to find the most similar cluster that best
represents them.

## Principal Component Analysis
```python
img_resized = cv2.resize(img, (300, 300))
# Flatten the image pixels
pixels = img_resized.reshape(-1, 3) # RGB
# Apply PCA to reduce to 2 or 3 dimensions for clustering
pca = PCA(n_components=2)
pixels_pca = pca.fit_transform(pixels)
```
<img width="612" height="584" alt="image" src="https://github.com/user-attachments/assets/5cb2a95b-ae34-4c99-8a40-27eadb1ee302" />


## Image Augmentation
```python
def load_and_augment(files, folder):
images = []
for filename in files:
img = cv2.imread(os.path.join(folder, filename))
img_resized = cv2.resize(img, (120, 120)) # Resize to fixed
size
augmented = [img_resized]
augmented.append(cv2.flip(img_resized, 1)) # Horizontal flip
augmented.append(cv2.flip(img_resized, 0)) # Vertical flip
augmented = [im.astype(np.float32)/255.0 for im in
augmented] # Normalize
```

## Training 
K-Means Clustering
```python
#K-means on reduced features
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(pixels_pca)
clustered = kmeans.labels_.reshape(img_resized.shape[:2])
```
<img width="468" height="503" alt="image" src="https://github.com/user-attachments/assets/3dd5d767-43ef-4eef-a468-25c56a007333" />
