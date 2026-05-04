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
