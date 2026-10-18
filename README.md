# Philadelphia Home Price Predictor

Hi there! Wondering what a home in Philadelphia is really worth? Enter a few details about the house and our tool estimates its market value in seconds.

We built it for real estate agents pricing a listing and for home buyers who want to check whether a price is fair.

**Live website:** https://philly-home-price-predictor.streamlit.app

Made by Owen, Hyeseong and Ian for BUSA 310 Business Analytics III (Group Project 1) at DePauw University.

## Team

| Name | LinkedIn / GitHub |
|---|---|
| Owen Nguyen | [LinkedIn](https://www.linkedin.com/in/owen-nguyen-work/) |
| Hyeseong Oh | [LinkedIn](https://www.linkedin.com/in/hyeseong-oh-722a712ba) · [GitHub](https://github.com/HyeseongO)  |
| Ian Porter | [LinkedIn](https://www.linkedin.com/in/ian-porter-312099337/) |

## Submission links

| Item | Link |
|---|---|
| Live website | https://philly-home-price-predictor.streamlit.app |
| Colab: data cleaning | [Open in Colab](https://colab.research.google.com/github/HyeseongO/philly-home-price-predictor/blob/main/notebook/02_data_cleaning.ipynb) |
| Colab: exploratory analysis | [Open in Colab](https://colab.research.google.com/github/HyeseongO/philly-home-price-predictor/blob/main/notebook/Group_Project.ipynb) |
| Colab: modeling | [Open in Colab](https://colab.research.google.com/github/HyeseongO/philly-home-price-predictor/blob/main/notebook/Group_Project_Modeling.ipynb) |
| Raw dataset | [Philadelphia Properties and Assessment History (OpenDataPhilly)](https://opendataphilly.org/datasets/philadelphia-properties-and-assessment-history) |
| Cleaned dataset | [`data/philly_sf_sales_clean.csv`](data/philly_sf_sales_clean.csv) |

## Data

| | |
|---|---|
| Source | City of Philadelphia, Office of Property Assessment (OPA), Properties dataset via OpenDataPhilly |
| Raw file | `opa_properties_public.csv`, downloaded 2026-10-08, 583,780 rows x 78 columns |
| Cleaning rules | Single-family homes only, sale date on or after 2024-01-01, sale price above $25,000 |
| Cleaned file | 37,469 sales, 80 columns (includes home coordinates `x_2272`, `y_2272`) |
| Sale dates | 2024-01-01 to 2026-09-02 |
| Modeling rows | 29,572 (bundled sales of several parcels at one price removed) |
| Privacy | Owner names and mailing addresses removed |

## Model

The website uses a Random Forest with ZIP and year fixed effects: ZIP code and sale year enter the model as categories, so each ZIP and each year gets its own price level. It was trained on a random 80/20 train/test split (`random_state=42`).
The full pipeline (missing value handling, category encoding and the model) is saved as one file, so the website applies the exact same steps as the notebook.

| | |
|---|---|
| Inputs | Living area, bedrooms, bathrooms, year built, stories, garage spaces, sale year, ZIP code, interior condition, exterior condition, central air, quality grade |
| Target | Sale price (US dollars) |
| Test performance | MAE $77,211 · RMSE $167,737 · R² 0.782 (5,915 test homes) |

Each estimate also shows a likely range: the middle 80% of how test homes with a similar estimate actually sold. On held-out test homes, the range contained the actual price 77.8% of the time.

The modeling notebook was last run in Colab with scikit-learn 1.9.1, the same version as the website's model file.

## How to reproduce the analysis

Run the notebooks in this order. In Google Colab, use **Runtime > Run all**.

1. `notebook/02_data_cleaning.ipynb`: raw OPA file to the cleaned dataset
2. `notebook/Group_Project.ipynb`: exploratory data analysis and charts
3. `notebook/Group_Project_Modeling.ipynb`: model comparison, evaluation, and the final website model with year and ZIP fixed effects (section 11)

The data cleaning notebook needs the raw OPA file: download it from OpenDataPhilly and upload it to `data/raw/` in Colab. The other notebooks load the cleaned dataset directly from this repository, so no upload is needed.

## How to run the website locally

Requires Python 3.13.

```bash
git clone https://github.com/HyeseongO/philly-home-price-predictor.git
cd philly-home-price-predictor
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

The site opens at http://localhost:8501.

**Test case:** ZIP 19143, 1,200 sq ft, 3 bedrooms, 1 bathroom, built 1925, 1 story, interior condition Rehabbed, exterior condition Average, quality grade C, garage and central air left as "Not sure". Expected result: **$202,077** (likely range $113,972 to $264,721).

## Project structure

```
app.py                  top navigation (entry point for Streamlit)
views/predict.py        prediction form and model
views/about.py          accuracy, data, limitations and interviews
views/team.py           team page
model/model.joblib      trained pipeline (scikit-learn 1.9.1)
model/input_options.json  valid dropdown values and numeric ranges
model/price_range.json  multipliers for the likely price range
notebook/               cleaning, EDA and modeling notebooks
data/                   cleaned dataset
assets/team/            team photos
requirements.txt        Python packages
```
