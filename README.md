# Laptop_Price_Prediction_ML: ML Workflow

This project follows a structured data science workflow, spanning from Exploratory Data Analysis (EDA) to predictive modeling of laptop prices. The dataset was was utilized as part of a university-led case study to practice end-to-end machine learning pipelines and regression analysis.

---

## Dataset Structure
The final processed dataset used for modeling contains the following core features:

| Feature | Type | Description |
| :--- | :--- | :--- |
| **Company** | Categorical | The laptop manufacturer (HP, Acer, Apple, etc.) |
| **Ram** | Numeric | System memory in GB |
| **SSD / HDD** | Numeric | Storage capacity in GB |
| **Inches** | Numeric | Screen size |
| **OpSys** | Categorical | Simplified Operating System (Windows, Mac, Other) |
| **Cpu_brand** | Categorical | Processor manufacturer and series |
| **Price** | Target | The final market price (Original Scale) |

---

## Project Workflow

### 1. Exploratory Data Analysis (EDA)

The initial phase focused on understanding the distribution and relationships within the dataset.

- **Structure Audit**  
  Used `df.info()` and `df.describe()` to inspect data types and statistical spreads.

- **Price Distribution**  
  Analyzed the `Price` column using `sns.histplot`.

- **Insight**  
  The price data was **right-skewed**. To improve model performance, a **Log Transformation** was applied (`np.log`), stabilizing the variance for the regression model.
<img width="701" height="532" alt="image" src="https://github.com/user-attachments/assets/85109e47-e843-45c9-a938-a9d07e86a85e" />

- **Feature Visualization**  
  Used scatterplots to visualize the relationship between `Price` and variables like `Ram`, `Inches`, `SSD`, and `HDD`.

- **Correlation Analysis**  
  Utilized `sns.heatmap` to identify linear relationships between numeric features.


### 2️. Data Preprocessing

Before modeling, the data was refined to ensure quality and compatibility:

- **Feature Selection**: Dropped high-cardinality or redundant string columns (`Product`, `Cpu`, `Gpu`) to focus on the extracted features.

- **Deduplication**: Cleaned the dataset by removing duplicate entries using `drop_duplicates()`.

- **OS Categorization**: Created a custom function to simplify the `OpSys` column into three clean categories: **Windows**, **Mac**, and **Other**.


### 3️. Feature Engineering & Encoding

- **Log Transformation**: Implemented a logarithmic scale on the target variable (`Price`) to create `LogPrice`, ensuring a more normal distribution for the Linear Regression algorithm.

- **One-Hot Encoding**: Converted categorical variables (`Company`, `Cpu_brand`, `Gpu_brand`, and `OpSys`) into dummy variables. The `drop_first=True` parameter was used to avoid the dummy variable trap (multicollinearity).


### 4️. Model Development (Linear Regression)
**Linear Regression** was chosed as the core algorithm to predict the price.
<img width="248" height="75" alt="image" src="https://github.com/user-attachments/assets/5865f9df-aacd-47e6-9fc7-c1a982d87c7e" />

*   **Data Splitting**: Partitioned the data into **Training** and **Testing** sets using `train_test_split` with a fixed `random_state=42`.
*   **Price Reversion**: Since the model was trained on `LogPrice`, the predictions were converted back to the original price scale using the exponential function `np.exp()` for the real price's accurate evaluation.

---

## Key Results
*   **Performance Metrics**: The model's accuracy was measured using **Mean Absolute Error (MAE)** and the **R-squared (R^2) Score**.
<img width="292" height="82" alt="image" src="https://github.com/user-attachments/assets/80ba4c9b-6850-4d07-861e-ed1fece49567" />

*   **Actual vs Predicted**: A scatterplot of predicted prices against actual prices shows strong linear alignment with a red striaght line representing the ideal `y=x` prediction.
<img width="1086" height="676" alt="image" src="https://github.com/user-attachments/assets/656d8a24-c9e6-4757-84b0-6f06fa23270e" />

*   **Skewness Correction**: The use of Log Transformation effectively handled the non-linear distribution of prices, resulting in a significantly stronger regression fit.
