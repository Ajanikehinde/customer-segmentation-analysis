# Customer Segmentation Analysis

A K-Means customer segmentation project using **350 e-commerce customers**, comparing multiple clustering strategies and evaluating whether the resulting segments represent genuine customer behaviour or underlying data structure.

> **Key finding:** The final clustering achieves a silhouette score of **0.7139**, but further analysis shows that most of the apparent segmentation is strongly associated with **City** and City-linked variables. This is an important data-quality finding and should be considered before using the segments for business decisions.

## Project Overview

This project demonstrates an end-to-end customer segmentation workflow:

* Data cleaning and quality checks
* Exploratory data analysis
* Feature preprocessing and standardisation
* K-Means clustering
* Silhouette-based model selection
* Customer segment profiling
* Comparative model evaluation
* Data-quality and confounding analysis
* Business-oriented interpretation of customer segments

The analysis compares three feature configurations against a baseline model before selecting the final numerical-only clustering approach.

## Dataset

`E-commerce_Customer_Behavior.csv` contains **350 customers and 11 variables**.

| Column                   | Type        | Description                          |
| ------------------------ | ----------- | ------------------------------------ |
| Customer ID              | Integer     | Identifier; removed before modelling |
| Gender                   | Categorical | Female / Male                        |
| Age                      | Numeric     | Customer age                         |
| City                     | Categorical | Customer city                        |
| Membership Type          | Categorical | Bronze / Silver / Gold               |
| Total Spend              | Numeric     | Customer spending                    |
| Items Purchased          | Numeric     | Number of items purchased            |
| Average Rating           | Numeric     | Average customer rating              |
| Discount Applied         | Boolean     | Whether a discount was applied       |
| Days Since Last Purchase | Numeric     | Recency indicator                    |
| Satisfaction Level       | Categorical | Unsatisfied / Neutral / Satisfied    |

The dataset contained **2 missing values in `Satisfaction Level`**, which were imputed using the column mode. No duplicate rows were identified.

## Methodology

### 1. Data Cleaning

* Loaded and inspected the dataset
* Checked data types, missing values and duplicates
* Imputed the two missing satisfaction values
* Removed `Customer ID` before modelling

### 2. Exploratory Data Analysis

The notebook examines:

* Numerical feature distributions
* Correlations between variables
* Categorical feature distributions
* Customer behaviour across demographic and purchasing variables

### 3. Baseline K-Means

A baseline model was created using the five numerical variables:

* Age
* Total Spend
* Items Purchased
* Average Rating
* Days Since Last Purchase

The variables were standardised using `StandardScaler`.

A fixed **K=4** was used as a simple reference model based on visual inspection of the elbow curve.

**Baseline silhouette score: 0.5626**

### 4. Model Comparison

Three feature configurations were evaluated across **K = 2 to 10**, with the optimal K selected using the maximum silhouette score.

| Model                      | Features                    | Selected K | Silhouette |
| -------------------------- | --------------------------- | ---------: | ---------: |
| Baseline                   | 5 numerical features        |          4 |     0.5626 |
| Full-feature               | Numerical + categorical     |          7 |     0.8029 |
| Behaviour-focused          | Full-feature excluding City |          8 |     0.7831 |
| **Numerical-only — Final** | **5 numerical features**    |      **7** | **0.7139** |

The numerical-only model was selected for the final segmentation because it provides a simpler and more interpretable feature space while avoiding direct use of categorical variables in K-Means.

## Final Customer Segments

The final model uses **K=7** and is based on the five standardised numerical features.

| Segment                         | Cluster |       Size |  Age | Total Spend | Items | Rating | Days Since Last Purchase |
| ------------------------------- | ------: | ---------: | ---: | ----------: | ----: | -----: | -----------------------: |
| High-Value Engaged Customers    |       2 | 58 (16.6%) | 29.1 |   $1,459.77 |  20.0 |   4.81 |                     11.2 |
| High-Value Satisfied Customers  |       5 | 59 (16.9%) | 30.7 |   $1,165.04 |  15.3 |   4.54 |                     24.6 |
| Active Mid-Value Customers      |       1 | 59 (16.9%) | 34.1 |     $805.49 |  11.7 |   4.17 |                     15.3 |
| At-Risk Average-Value Customers |       3 |  34 (9.7%) | 26.8 |     $703.69 |  12.8 |   4.02 |                     53.2 |
| Low-Value At-Risk Customers     |       6 |  24 (6.9%) | 32.0 |     $671.55 |  10.0 |   3.80 |                     34.6 |
| Low-Value Inactive Customers    |       4 | 58 (16.6%) | 42.0 |     $499.88 |   9.4 |   3.46 |                     40.5 |
| Low-Value Neutral Customers     |       0 | 58 (16.6%) | 36.7 |     $446.89 |   7.6 |   3.19 |                     22.8 |

The segment labels are descriptive summaries of the observed cluster profiles rather than independently validated customer personas.

## Key Data-Quality Finding: City Confounding

The clustering results require an important qualification.

Although **City was excluded from the final numerical-only model**, the analysis found that several numerical variables are strongly structured by City:

* `Total Spend`, `Items Purchased`, and `Average Rating` occupy **non-overlapping numeric bands across cities**.
* `Membership Type` is a **deterministic function of City** in this dataset, with zero exceptions across all 350 records.
* **6 of the 7 final clusters** map one-to-one onto combinations of `City`, `Membership Type`, and `Discount Applied`.
* The remaining split occurs within the Miami / Silver / discount-applied group and is primarily associated with **Age and recency**.

This means the high silhouette scores should **not** be interpreted as evidence that seven independent behavioural personas have been discovered.

A more accurate interpretation is:

> **The segmentation primarily reflects city-linked customer tiers, with one additional age/recency-based split within a Miami cohort.**

The non-overlapping numerical ranges also suggest that the dataset may be synthetic or template-generated rather than representative of naturally collected transaction data.

**Practical implication:** before applying this methodology to a real customer dataset, the City-confounding checks should be repeated to confirm that the numerical variables are not simply encoding another categorical variable.

## Limitations

### Mixed-type clustering

The full-feature and behaviour-focused models combine one-hot-encoded categorical variables with standardised numerical variables under Euclidean-distance K-Means.

For genuinely mixed-type customer data, approaches such as **K-Prototypes** or **Gower-distance-based clustering** may provide a more appropriate methodology.

### Dataset size

The dataset contains only **350 customers**. It is suitable for demonstrating the methodology but is too small to support broad generalisation.

### Data quality

The strong City-linked structure limits the extent to which the final clusters can be interpreted as independent behavioural segments.

### Imputation

Only two `Satisfaction Level` values were missing and were replaced using the mode. The impact is therefore limited, but the imputation should still be documented.

## Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Seaborn
* Jupyter Notebook
* K-Means clustering
* StandardScaler
* Silhouette analysis

## Reproducing the Analysis

Install the required packages:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

The notebook was verified to execute sequentially from **execution count 1 to 81 with no missing execution counts**.

Open the notebook with:

```bash
jupyter notebook Customer_Segmentation_final.ipynb
```

Then select:

**Kernel → Restart & Run All**

## Repository Structure

```text
customer-segmentation-analysis/
└── Customer_Segmentation_final.ipynb
```

## Project Takeaway

This project demonstrates that a strong clustering score alone is not sufficient to establish meaningful customer segments.

The analysis combines **model evaluation with data-quality investigation**, showing why segmentation results should be checked for hidden relationships and confounding variables before being translated into business decisions.
