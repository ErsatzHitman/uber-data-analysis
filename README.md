# Uber Surge Pricing Analysis

A comprehensive Python script for analyzing Uber ride data to understand and visualize surge pricing patterns. This tool can load your own Uber data CSV or generate a realistic sample dataset for immediate analysis.

## Overview

This project provides a complete, end-to-end pipeline for surge pricing analysis within a single Python class, `UberDataAnalyzer`. It handles everything from data loading and cleaning to advanced feature engineering, exploratory data analysis (EDA), insight generation, and exporting a clean dataset for tools like Tableau.

The primary goal is to identify the key drivers of surge pricing, such as time of day, day of the week, weather, and ride demand.

## Key Features

  * **Sample Data Generation:** If no CSV file is provided, the script automatically generates 10,000 sample ride records to run the full analysis.
  * **Data Cleaning:** Handles missing values and removes unrealistic data points (e.g., negative fares, zero-mile trips).
  * **Feature Engineering:** Creates new, insightful columns:
      * `time_of_day` (Night, Morning, Afternoon, Evening)
      * `is_weekend` (True/False)
      * `surge_category` (No Surge, Low, Medium, High)
      * `demand_score` (A custom score based on peak hours, weekends, and precipitation)
  * **Comprehensive EDA:** Generates a 9-plot dashboard to visualize key relationships.
  * **Text-Based Insights:** Prints a summary of key findings, including peak surge hours, weather impact, and weekend vs. weekday trends.
  * **Tableau Export:** Saves a cleaned and feature-rich CSV file (`uber_data_tableau_ready.csv`) for external visualization.

## Getting Started

### Prerequisites

You must have Python 3 and the following libraries installed:

  * pandas
  * numpy
  * matplotlib
  * seaborn

### Installation

1.  **Clone the repository (or download the script):**

    ```bash
    git clone https://your-repo-url/uber-analysis.git
    cd uber-analysis
    ```

2.  **Install the required libraries:**

    ```bash
    pip install pandas numpy matplotlib seaborn
    ```

### Running the Analysis

There are two ways to run the script:

**1. Run with Auto-Generated Sample Data (Easiest)**

Simply execute the Python script. It will detect that no data file is present and generate its own.

```bash
python uber_analysis.py
```

**2. Run with Your Own Data**

1.  Place your `your_data.csv` file in the same directory.

2.  Modify the `main()` function at the bottom of the script to pass your file's name:

    ```python
    def main():
        """
        Main function to run the Uber data analysis
        """
        # Initialize analyzer
        analyzer = UberDataAnalyzer()
        
        # Run complete analysis with YOUR data
        analyzer.run_complete_analysis('path/to/your/uber_data.csv')

    if __name__ == "__main__":
        main()
    ```

3.  Run the script:

    ```bash
    python uber_analysis.py
    ```

## Analysis & Visualizations

The script performs a detailed analysis and generates the following key outputs:

### 1\. EDA Dashboard (Matplotlib Plot)

A 3x3 plot window opens, showing:

1.  **Surge Multiplier Distribution:** A bar chart showing the frequency of each surge level (1.0x, 1.5x, etc.).
2.  **Average Surge by Hour:** A line plot tracking the average surge multiplier for each hour of the day.
3.  **Average Surge by Day of Week:** A bar chart comparing surge levels across the week.
4.  **Fare Amount vs. Distance:** A scatter plot colored by surge multiplier to see how surge affects fare.
5.  **Average Surge by Time of Day:** A bar chart for 'Morning', 'Afternoon', 'Evening', and 'Night'.
6.  **Correlation Heatmap:** A heatmap showing correlations between `surge_multiplier`, `fare_amount`, `distance_miles`, `precipitation`, and `demand_score`.
7.  **Ride Demand by Hour:** A line plot showing the total number of rides requested per hour.
8.  **Surge vs. Precipitation:** A bar chart showing how increasing precipitation affects the average surge.
9.  **Average Surge by Vehicle Type:** A bar chart comparing surge across `UberX`, `UberBlack`, `UberXL`, etc.

### 2\. Console Summary Report

A detailed summary is printed directly to your console, highlighting:

  * **Key Metrics:** Total rides, average surge, and percentage of rides with surge.
  * **Top 3 Peak Surge Hours:** Identifies the most expensive hours to ride.
  * **Weekend vs. Weekday Surge:** A direct comparison of average surge multipliers.
  * **Weather Impact:** The percentage increase in surge during high precipitation.
  * **Business Insights & Recommendations:** Actionable insights based on the data.

## Code Structure

The analysis is encapsulated in the `UberDataAnalyzer` class.

  * `__init__(self, data_path=None)`: Initializes the class.
  * `load_data(self, data_path)`: Loads data from a specified CSV path.
  * `generate_sample_data(self)`: Creates a 10,000-record sample DataFrame if no data is loaded.
  * `data_cleaning(self)`: Cleans data and engineers new features.
  * `exploratory_data_analysis(self)`: Generates and displays the 9-plot `matplotlib` dashboard.
  * `surge_pricing_insights(self)`: Analyzes the processed data to find and print key insights.
  * `_generate_summary_report(self)`: A private helper method to format the final text report.
  * `prepare_tableau_data(self, output_path=...)`: Exports the final, clean DataFrame to a CSV.
  * `run_complete_analysis(self, data_path=None)`: A single public method that runs the entire pipeline in the correct order.

## Output File

  * **`uber_data_tableau_ready.csv`**: A cleaned, processed CSV file with all original and engineered features, ready for import into Tableau or other BI tools.
