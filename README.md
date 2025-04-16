# Exploratory-Data-Analysis---Customer-Loans-in-Finance

This repository contains the source code and documentation for a data science project focused on understanding loan repayment behaviors, identifying risks, and analyzing customer loan data from a financial institution. The project uses data extracted from a cloud-based PostgreSQL database and applies advanced data transformation, exploratory data analysis (EDA), and visualization techniques to uncover insights into repayment status, charged-off losses, and at-risk customers.

## Table of Contents

- [Project Overview](#project-overview)
- [Objectives](#objectives)
- [Environment Setup](#environment-setup)
- [Project Structure](#project-structure)
- [Data Extraction](#data-extraction)
- [Data Dictionary](#data-dictionary)
- [EDA Workflow](#eda-workflow)
- [Analysis Performed](#analysis-performed)
- [How to Run](#how-to-run)
- [Technologies Used](#technologies-used)
- [License](#license)

## Project Overview

This project simulates the role of a data analyst at a financial institution. The goal is to gain a comprehensive understanding of loan performance and repayment risk through structured EDA and loss estimation techniques. It follows a milestone-based development approach and builds progressively through data extraction, transformation, and insight generation.

## Objectives

- Extract and inspect loan data from a PostgreSQL RDS instance.
- Handle and clean missing or inconsistent data.
- Convert categorical or string-based columns into usable formats.
- Visualize the distribution and skewness of data.
- Apply various data transformations to normalize skewed data.
- Identify and remove outliers and highly correlated columns.
- Estimate actual and projected financial losses due to charged-off and late loans.
- Identify user segments with high loss risk and provide actionable recommendations.

## Environment Setup

1. Clone the repository to your local system:
   ```
   git clone https://github.com/yourusername/loan-eda-project.git
   cd loan-eda-project
   ```

2. Create and activate a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   ```

3. Install the dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Create a `credentials.yaml` file in the root directory containing your RDS credentials:
   ```yaml
   RDS_HOST: your-rds-endpoint
   RDS_USER: your-db-username
   RDS_PASSWORD: your-db-password
   RDS_DATABASE: your-db-name
   RDS_PORT: 5432
   ```

## Project Structure

```
.
├── db_utils.py                # Database connector for RDS
├── extract_save_data.py      # CLI script to pull and save loan data
├── transformation_plotting.py # Classes for data cleaning, transformation, and plotting
├── credentials.yaml           # YAML file with RDS credentials (not pushed to GitHub)
├── loan_payments.csv         # Extracted loan data (optional local storage)
├── README.md                  # Project documentation
```

## Data Extraction

The `RDSDatabaseConnector` class (in `db_utils.py`) initializes a connection to a PostgreSQL RDS instance using credentials from the `credentials.yaml` file. The script `extract_save_data.py` is responsible for:

- Reading credentials from the YAML file.
- Connecting to the RDS instance.
- Extracting the `loan_payments` table.
- Saving the dataset as a CSV file for local processing.

## Data Dictionary

A selection of relevant columns in the dataset includes:

- `id`: Unique loan identifier
- `member_id`: ID of the loan applicant
- `loan_amount`: Loan amount requested
- `funded_amount`: Amount funded
- `term`: Duration of the loan (in months)
- `int_rate`: Interest rate
- `installment`: Monthly payment
- `grade`, `sub_grade`: Credit grade
- `employment_length`: Duration of employment
- `home_ownership`, `annual_inc`: Borrower income details
- `verification_status`: Verification status
- `loan_status`: Current repayment status
- `total_payment`: Total expected repayment amount
- `total_rec_prncp`, `total_rec_int`, `recoveries`, `late_fees`: Recovered components
- `application_type`: Individual vs joint loan application

See the full dictionary in the project milestone instructions or in the notebook.

## EDA Workflow

The `transformation_plotting.py` file encapsulates the core logic across several modular classes:

### Data Preparation
- `DataTransform`: Converts columns to appropriate formats (e.g., `term` to int).
- `DataFrameTransform`: Imputes missing values, removes outliers, transforms skewed data.

### Analysis & Transformation
- Skew correction using log, cube root, Box-Cox, Yeo-Johnson, and other transformations.
- Removal of columns with high null percentages or constant values.
- Dropping of highly correlated columns.
- Detection and visualization of strand outliers.

### Visualization
- Heatmaps for null values and correlations.
- Histograms and Q-Q plots for transformations.
- Repayment vs loss visual summaries.
- Grouped bar plots for categorical variable comparisons.

## Analysis Performed

The `LoanAnalyzer` and `Plotter` classes are used to derive and visualize insights such as:

- Current Repayment Summary: Percentage of total repayment recovered to date.
- 6-Month Projection: Forecast of repayment over the next 6 months.
- Charged-Off Loss: Analysis of fully defaulted loans.
- Projected Loss: What losses would be if all charged-off loans were repaid.
- At-Risk Users: Estimating losses for currently late payers.
- High Risk Tagging: Loans flagged for potential financial loss.
- Categorical Risk Indicators: Comparing `grade`, `home_ownership`, and `purpose` between good and bad loans.

## How to Run

To extract the data:

```
python extract_save_data.py
```

To perform EDA and visual analysis, import and use the classes in a Jupyter Notebook or script:

```python
from transformation_plotting import DataFrameTransform, Plotter, LoanAnalyzer
```

Follow the milestone-based logic for analysis and plotting.

## Technologies Used

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- statsmodels
- plotly
- SQLAlchemy
- psycopg2
- PyYAML

## License

This project is provided for educational purposes. No commercial use is intended. Please credit the author(s) if you use this code as a foundation for your own work.
