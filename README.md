# Customer Segmentation Analysis

K-Means clustering on 350 e-commerce customers, comparing four modeling approaches and profiling the resulting segments — with an explicit check for a data-quality issue that undermines most of the "behavioural" story the segments appear to tell.

**Read the [Key Caveat](#key-caveat-most-of-this-is-city-not-behaviour) section before using any conclusion from this project in a real decision.**

## Contents

```
.
├── Customer_Segmentation_corrected.ipynb   # main analysis notebook
├── data/
│   └── E-commerce_Customer_Behavior.csv    # source dataset (350 rows, 11 columns)
└── README.md
```

## Dataset

`E-commerce_Customer_Behavior.csv` — 350 customers, 11 columns:

| Column | Type | Notes |
|---|---|---|
| Customer ID | int | dropped before modeling |
| Gender | categorical | Female / Male |
| Age | numeric | |
| City | categorical | 6 cities |
| Membership Type | categorical | Bronze / Silver / Gold |
| Total Spend | numeric | |
| Items Purchased | numeric | |
| Average Rating | numeric | |
| Discount Applied | boolean | |
| Days Since Last Purchase | numeric | used as a recency signal |
| Satisfaction Level | categorical | Unsatisfied / Neutral / Satisfied — 2 missing values, imputed with the mode |

No duplicate rows.

## Methodology

The notebook is organized into the following sections:

1. **Data Loading & Cleaning** — load, check dtypes/nulls/duplicates, impute the 2 missing `Satisfaction Level` values, drop `Customer ID`.
2. **Exploratory Data Analysis** — distributions, correlations, categorical breakdowns.
3. **Baseline K-Means Clustering** — standardize the 5 numeric features, fit `K=4` (chosen by eye from the elbow curve) as a naive reference point.
4. **Improved Clustering Methodology** — three feature configurations, each evaluated across `K = 2..10` and selected by max silhouette score:
   - **Full-feature**: 5 numeric + one-hot-encoded categoricals
   - **Behaviour-focused**: full-feature minus `City`
   - **Numerical-only**: the 5 numeric features alone
5. **Final Model** — the numerical-only configuration is selected, profiled, and its 7 clusters are assigned descriptive business labels.
   - **5a. Checking for a City Confound** — tests whether the numeric features are actually independent of `City`. (They aren't — see below.)
6. **Segment Comparison Visualisations** — bar charts and cross-tabs across the 7 final segments.
7. **Additional Bivariate Relationships** — supplementary scatter plots.

### Model comparison

| Model | Feature set | Selected K | Silhouette score |
|---|---|---|---|
| Baseline | 5 numeric (standardized) | 4 *(fixed, not optimized)* | 0.5626 |
| Full-feature | numeric + one-hot categoricals | 7 | 0.8029 |
| Behaviour-focused | full-feature minus `City` | 8 | 0.7831 |
| **Numerical-only (final model)** | 5 numeric only | **7** | **0.7139** |

`K` for every non-baseline model is chosen by `argmax` on silhouette score — not hand-picked.

## Results: the 7 customer segments

Numerical-only K-Means, `K=7`, ordered by average `Total Spend`:

| Segment | Cluster ID | Size | Age | Total Spend | Items Purchased | Avg. Rating | Days Since Last Purchase | Primary City | Membership | Satisfaction |
|---|---|---|---|---|---|---|---|---|---|---|
| High-Value Engaged Customers | 2 | 58 (16.6%) | 29.1 | $1,459.77 | 20.0 | 4.81 | 11.2 | San Francisco | Gold | Satisfied |
| High-Value Satisfied Customers | 5 | 59 (16.9%) | 30.7 | $1,165.04 | 15.3 | 4.54 | 24.6 | New York | Gold | Satisfied |
| Active Mid-Value Customers | 1 | 59 (16.9%) | 34.1 | $805.49 | 11.7 | 4.17 | 15.3 | Los Angeles | Silver | Neutral |
| At-Risk Average-Value Customers | 3 | 34 (9.7%) | 26.8 | $703.69 | 12.8 | 4.02 | 53.2 | Miami | Silver | Unsatisfied |
| Low-Value At-Risk Customers | 6 | 24 (6.9%) | 32.0 | $671.55 | 10.0 | 3.80 | 34.6 | Miami | Silver | Unsatisfied |
| Low-Value Inactive Customers | 4 | 58 (16.6%) | 42.0 | $499.88 | 9.4 | 3.46 | 40.5 | Chicago | Bronze | Unsatisfied |
| Low-Value Neutral Customers | 0 | 58 (16.6%) | 36.7 | $446.89 | 7.6 | 3.19 | 22.8 | Houston | Bronze | Neutral |

*"At-Risk Average-Value Customers" (cluster 3) has the highest `Days Since Last Purchase` of all seven segments (53.2 days vs. an overall mean of 26.6) and below-average spend — this label was corrected from an earlier, self-contradicting draft that called this group "Active."*

## Key caveat: most of this is City, not behaviour

Section 5a of the notebook checks something the silhouette scores above should have raised on their own: **0.71–0.80 is an unusually high silhouette score for real customer behavioural data.** The check finds why:

- `Total Spend`, `Items Purchased`, and `Average Rating` fall into **non-overlapping numeric bands by `City`**, with real gaps between cities (e.g. `Total Spend` jumps from a max of $530 in the Bronze-tier cities straight to a min of $660 in the next tier — no real, continuously-distributed customer population looks like that).
- `Membership Type` is a **deterministic function of `City`**, with zero exceptions across all 350 rows.
- Of the 7 final segments, **6 map exactly one-to-one onto a `City` × `Membership Type` × `Discount Applied` combination.** The clustering was run on 5 numeric columns with `City` excluded from the feature set entirely — and it reconstructed `City` anyway.
- Only **one** segment split reflects genuine within-city structure: the Miami / Silver / Discount-applied cohort splits into two segments along `Age` and recency.

**Practical takeaway:** treat this segmentation as *"city tier, plus one age/recency-based split within a single city,"* not as seven independently discovered behavioural personas. The non-overlapping numeric bands are also a strong signal this dataset is synthetic or template-generated rather than raw transaction logs — if you're using this repo as a template for a real dataset, re-run Section 5a first and confirm it comes back clean before trusting the segment profiles.

## Other limitations

- **Mixed-type clustering.** The full-feature and behaviour-focused models combine one-hot-encoded categorical dummies with standardized continuous features under plain Euclidean-distance K-Means, which can let the dummy columns dominate cluster formation. [K-Prototypes](https://github.com/nicodv/kmodes) or a Gower-distance-based approach would be more defensible for genuinely mixed-type data.
- **Small sample.** n = 350. Fine for demonstrating a methodology; too small and too clean (see above) to generalize from.
- **Imputation.** 2 of 350 `Satisfaction Level` values were filled with the column mode — low impact given the count, but worth knowing before extending the pipeline.

## Setup

```bash
pip install pandas numpy scikit-learn matplotlib seaborn jupyter
```

Verified working with:

| Package | Version |
|---|---|
| pandas | 3.0.2 |
| numpy | 2.4.4 |
| scikit-learn | 1.8.0 |
| matplotlib | 3.10.8 |
| seaborn | 0.13.2 |

(Older pandas 2.x / earlier scikit-learn versions should also work — nothing in the notebook depends on pandas 3.x-specific behavior.)

## Running the notebook

```bash
jupyter notebook Customer_Segmentation_corrected.ipynb
```

Run **Kernel → Restart & Run All**. The notebook is verified to execute top-to-bottom with zero errors and sequential execution counts (1 → 81) against the dataset in `data/`.

## Revision notes

This notebook was corrected from an earlier draft that would not execute cleanly:

- Removed two dead cells that referenced undefined variables (`numerical_columns` / `categorical_columns`) — leftovers from an earlier, discarded preprocessing approach, causing `NameError` on a clean run.
- Fixed the behaviour-focused model's `K` selection to use the same `argmax`-on-silhouette rule as the full-feature model, instead of a hardcoded, unjustified `K=7`. (It now correctly selects `K=8`.)
- Corrected the "At-Risk Active-Value Customers" segment label, which contradicted its own underlying statistics, to "At-Risk Average-Value Customers."
- Added Section 5a, which tests for and discloses the City confound described above.
