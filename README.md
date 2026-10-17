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
| Colab: modeling | [Open in Colab](https://colab.research.google.com/github/HyeseongO/philly-home-price-predictor/blob/main/notebook/Group_Project_Modeling.ipynb) | |
| Raw dataset | [Philadelphia Properties and Assessment History (OpenDataPhilly)](https://opendataphilly.org/datasets/philadelphia-properties-and-assessment-history) |
| Cleaned dataset | [`data/philly_sf_sales_clean.csv`](data/philly_sf_sales_clean.csv) |

## Data

| | |
|---|---|
| Source | City of Philadelphia, Office of Property Assessment (OPA), Properties dataset via OpenDataPhilly |
| Raw file | `opa_properties_public.csv`, downloaded 2026-10-08, 583,780 rows x 78 columns |
| Cleaning rules | Single-family homes only, sale date on or after 2024-01-01, sale price above $25,000 |
| Cleaned file | 37,469 sales, 78 columns |
| Sale dates | 2024-01-01 to 2026-09-02 |
| Modeling rows | 29,572 (bundled sales of several parcels at one price removed) |
| Privacy | Owner names and mailing addresses removed |

## Model

The website currently uses a Random Forest trained on a random 80/20 train/test split (`random_state=42`).
The full pipeline (missing value handling, category encoding and the model) is saved as one file, so the website applies the exact same steps as the notebook.

| | |
|---|---|
| Inputs | Living area, bedrooms, bathrooms, year built, stories, garage spaces, sale year, ZIP code, interior condition, exterior condition, central air, quality grade |
| Target | Sale price (US dollars) |
| Test performance | MAE $77,149 · RMSE $167,630 · R² 0.782 (5,915 test homes) |

The modeling notebook was last run in Colab with scikit-learn 1.6.1. The website's model file was saved with scikit-learn 1.9.1, and both give identical predictions.

## How to reproduce the analysis

Run the notebooks in this order. In Google Colab, use **Runtime > Run all**.

1. `notebook/02_data_cleaning.ipynb`: raw OPA file to the cleaned dataset
2. `notebook/Group_Project.ipynb`: exploratory data analysis and charts
3. `notebook/Group_Project_Modeling.ipynb`: model comparison, evaluation and saving the final model

TODO: note where each notebook expects its input file (for example, upload the CSV in Colab or load it from this repository).

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

**Test case:** ZIP 19143, 1,200 sq ft, 3 bedrooms, 1 bathroom, built 1925, 1 story, interior condition Rehabbed, exterior condition Average, quality grade C, garage and central air left as "Not sure". Expected result: **$203,520**.

## Project structure

```
app.py                  top navigation (entry point for Streamlit)
views/predict.py        prediction form and model
views/team.py           team page
model/model.joblib      trained pipeline (scikit-learn 1.9.1)
model/input_options.json  valid dropdown values and numeric ranges
notebook/               cleaning, EDA and modeling notebooks
data/                   cleaned dataset
assets/team/            team photos
requirements.txt        Python packages
```
