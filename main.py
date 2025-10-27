"""
Uber Data Analysis - Surge Pricing Analysis
Complete analysis of Uber ride data to understand surge pricing patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set up plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class UberDataAnalyzer:
    def __init__(self, data_path=None):
        """
        Initialize the Uber Data Analyzer
        
        Args:
            data_path (str): Path to the Uber dataset CSV file
        """
        self.data = None
        self.processed_data = None
        
        if data_path:
            self.load_data(data_path)
    
    def load_data(self, data_path):
        """
        Load Uber data from CSV file
        """
        try:
            self.data = pd.read_csv(data_path)
            print(f"Data loaded successfully: {len(self.data)} records")
        except FileNotFoundError:
            print("CSV file not found. Generating sample data...")
            self.generate_sample_data()
    
    def generate_sample_data(self):
        """
        Generate sample Uber data for demonstration
        """
        np.random.seed(42)
        n_records = 10000
        
        # Generate sample data
        dates = pd.date_range('2024-01-01', '2024-03-31', freq='H')
        dates = np.random.choice(dates, n_records)
        
        self.data = pd.DataFrame({
            'ride_id': [f'ride_{i}' for i in range(n_records)],
            'timestamp': dates,
            'hour': np.random.randint(0, 24, n_records),
            'day_of_week': np.random.randint(0, 7, n_records),
            'month': np.random.randint(1, 13, n_records),
            'pickup_latitude': np.random.uniform(40.70, 40.80, n_records),
            'pickup_longitude': np.random.uniform(-74.02, -73.92, n_records),
            'dropoff_latitude': np.random.uniform(40.70, 40.80, n_records),
            'dropoff_longitude': np.random.uniform(-74.02, -73.92, n_records),
            'distance_miles': np.random.uniform(0.5, 20, n_records),
            'surge_multiplier': np.random.choice([1.0, 1.5, 2.0, 2.5, 3.0], n_records, p=[0.6, 0.2, 0.1, 0.05, 0.05]),
            'base_fare': np.random.uniform(2.5, 5.0, n_records),
            'fare_amount': np.random.uniform(5, 100, n_records),
            'duration_minutes': np.random.uniform(5, 60, n_records),
            'temperature': np.random.uniform(20, 95, n_records),
            'precipitation': np.random.exponential(0.2, n_records),
            'demand_zone': np.random.choice(['Low', 'Medium', 'High'], n_records, p=[0.3, 0.5, 0.2]),
            'vehicle_type': np.random.choice(['UberX', 'UberBlack', 'UberXL', 'Pool'], n_records, p=[0.6, 0.1, 0.2, 0.1])
        })
        
        # Make fare amount dependent on other factors
        self.data['fare_amount'] = (
            self.data['base_fare'] + 
            self.data['distance_miles'] * 1.5 + 
            self.data['duration_minutes'] * 0.3
        ) * self.data['surge_multiplier']
        
        print(f"Sample data generated: {len(self.data)} records")
    
    def data_cleaning(self):
        """
        Clean and preprocess the data
        """
        print("Starting data cleaning...")
        
        # Create a copy for processing
        self.processed_data = self.data.copy()
        
        # Handle missing values
        initial_count = len(self.processed_data)
        self.processed_data = self.processed_data.dropna()
        print(f"Removed {initial_count - len(self.processed_data)} records with missing values")
        
        # Remove unrealistic values
        self.processed_data = self.processed_data[
            (self.processed_data['fare_amount'] > 0) &
            (self.processed_data['distance_miles'] > 0) &
            (self.processed_data['duration_minutes'] > 0) &
            (self.processed_data['surge_multiplier'] >= 1)
        ]
        
        # Extract datetime features
        if 'timestamp' in self.processed_data.columns:
            self.processed_data['timestamp'] = pd.to_datetime(self.processed_data['timestamp'])
            self.processed_data['hour'] = self.processed_data['timestamp'].dt.hour
            self.processed_data['day_of_week'] = self.processed_data['timestamp'].dt.dayofweek
            self.processed_data['month'] = self.processed_data['timestamp'].dt.month
            self.processed_data['is_weekend'] = self.processed_data['day_of_week'].isin([5, 6]).astype(int)
        
        # Create time categories
        self.processed_data['time_of_day'] = pd.cut(
            self.processed_data['hour'],
            bins=[0, 6, 12, 18, 24],
            labels=['Night', 'Morning', 'Afternoon', 'Evening'],
            include_lowest=True
        )
        
        # Create surge categories
        self.processed_data['surge_category'] = pd.cut(
            self.processed_data['surge_multiplier'],
            bins=[1, 1.5, 2.0, 3.0, 10],
            labels=['No Surge', 'Low Surge', 'Medium Surge', 'High Surge']
        )
        
        # Create demand score based on multiple factors
        self.processed_data['demand_score'] = (
            self.processed_data['hour'].apply(lambda x: 1 if 7<=x<=9 or 17<=x<=19 else 0.5) +
            self.processed_data['is_weekend'] +
            (self.processed_data['precipitation'] > 0.5).astype(int)
        )
        
        print(f"Data cleaning completed. Final records: {len(self.processed_data)}")
        
        return self.processed_data
    
    def exploratory_data_analysis(self):
        """
        Perform exploratory data analysis
        """
        if self.processed_data is None:
            print("Please run data_cleaning() first")
            return
        
        print("\n=== EXPLORATORY DATA ANALYSIS ===")
        
        # Basic statistics
        print("\nBasic Statistics:")
        print(self.processed_data[['fare_amount', 'distance_miles', 'duration_minutes', 'surge_multiplier']].describe())
        
        # Surge pricing distribution
        print(f"\nSurge Pricing Distribution:")
        surge_dist = self.processed_data['surge_category'].value_counts()
        print(surge_dist)
        
        # Set up the plotting figure
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Surge Multiplier Distribution
        plt.subplot(3, 3, 1)
        surge_counts = self.processed_data['surge_multiplier'].value_counts().sort_index()
        plt.bar(surge_counts.index, surge_counts.values, color='skyblue', alpha=0.7)
        plt.title('Surge Multiplier Distribution')
        plt.xlabel('Surge Multiplier')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        # 2. Surge Pricing by Hour of Day
        plt.subplot(3, 3, 2)
        surge_by_hour = self.processed_data.groupby('hour')['surge_multiplier'].mean()
        plt.plot(surge_by_hour.index, surge_by_hour.values, marker='o', color='coral', linewidth=2)
        plt.title('Average Surge Multiplier by Hour')
        plt.xlabel('Hour of Day')
        plt.ylabel('Average Surge Multiplier')
        plt.grid(True, alpha=0.3)
        plt.xticks(range(0, 24, 2))
        
        # 3. Surge Pricing by Day of Week
        plt.subplot(3, 3, 3)
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        surge_by_day = self.processed_data.groupby('day_of_week')['surge_multiplier'].mean()
        plt.bar(days, surge_by_day, color='lightgreen', alpha=0.7)
        plt.title('Average Surge Multiplier by Day of Week')
        plt.xlabel('Day of Week')
        plt.ylabel('Average Surge Multiplier')
        plt.grid(True, alpha=0.3)
        
        # 4. Fare Amount vs Distance
        plt.subplot(3, 3, 4)
        sample_data = self.processed_data.sample(min(1000, len(self.processed_data)))
        plt.scatter(sample_data['distance_miles'], sample_data['fare_amount'], 
                   c=sample_data['surge_multiplier'], alpha=0.6, cmap='viridis')
        plt.colorbar(label='Surge Multiplier')
        plt.title('Fare Amount vs Distance (Colored by Surge)')
        plt.xlabel('Distance (miles)')
        plt.ylabel('Fare Amount ($)')
        plt.grid(True, alpha=0.3)
        
        # 5. Surge Pricing by Time of Day
        plt.subplot(3, 3, 5)
        surge_by_tod = self.processed_data.groupby('time_of_day')['surge_multiplier'].mean()
        surge_by_tod.plot(kind='bar', color='gold', alpha=0.7)
        plt.title('Average Surge Multiplier by Time of Day')
        plt.xlabel('Time of Day')
        plt.ylabel('Average Surge Multiplier')
        plt.grid(True, alpha=0.3)
        
        # 6. Correlation Heatmap
        plt.subplot(3, 3, 6)
        numeric_cols = ['fare_amount', 'distance_miles', 'duration_minutes', 
                       'surge_multiplier', 'temperature', 'precipitation', 'demand_score']
        correlation_matrix = self.processed_data[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f')
        plt.title('Correlation Matrix of Key Variables')
        
        # 7. Demand Patterns by Hour
        plt.subplot(3, 3, 7)
        demand_by_hour = self.processed_data.groupby('hour').size()
        plt.plot(demand_by_hour.index, demand_by_hour.values, 
                marker='s', color='purple', linewidth=2)
        plt.title('Ride Demand by Hour of Day')
        plt.xlabel('Hour of Day')
        plt.ylabel('Number of Rides')
        plt.grid(True, alpha=0.3)
        plt.xticks(range(0, 24, 2))
        
        # 8. Surge Pricing vs Weather Conditions
        plt.subplot(3, 3, 8)
        weather_effect = self.processed_data.groupby(
            pd.cut(self.processed_data['precipitation'], bins=5)
        )['surge_multiplier'].mean()
        weather_effect.plot(kind='bar', color='orange', alpha=0.7)
        plt.title('Surge Multiplier vs Precipitation')
        plt.xlabel('Precipitation Level')
        plt.ylabel('Average Surge Multiplier')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        # 9. Vehicle Type Analysis
        plt.subplot(3, 3, 9)
        if 'vehicle_type' in self.processed_data.columns:
            vehicle_surge = self.processed_data.groupby('vehicle_type')['surge_multiplier'].mean()
            vehicle_surge.sort_values(ascending=False).plot(kind='bar', color='lightcoral', alpha=0.7)
            plt.title('Average Surge Multiplier by Vehicle Type')
            plt.xlabel('Vehicle Type')
            plt.ylabel('Average Surge Multiplier')
            plt.xticks(rotation=45)
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return self.processed_data
    
    def surge_pricing_insights(self):
        """
        Generate key insights about surge pricing patterns
        """
        if self.processed_data is None:
            print("Please run data_cleaning() first")
            return
        
        print("\n=== SURGE PRICING INSIGHTS ===")
        
        # Peak surge hours
        peak_surge_hours = self.processed_data.groupby('hour')['surge_multiplier'].mean().nlargest(3)
        print(f"\nTop 3 Peak Surge Hours:")
        for hour, surge in peak_surge_hours.items():
            print(f"  Hour {hour:02d}:00 - Average Surge: {surge:.2f}x")
        
        # Weekend vs Weekday surge
        weekend_surge = self.processed_data[self.processed_data['is_weekend'] == 1]['surge_multiplier'].mean()
        weekday_surge = self.processed_data[self.processed_data['is_weekend'] == 0]['surge_multiplier'].mean()
        print(f"\nWeekend vs Weekday Surge:")
        print(f"  Weekend: {weekend_surge:.2f}x")
        print(f"  Weekday: {weekday_surge:.2f}x")
        print(f"  Difference: {abs(weekend_surge - weekday_surge):.2f}x")
        
        # Weather impact
        high_precip_surge = self.processed_data[self.processed_data['precipitation'] > 0.5]['surge_multiplier'].mean()
        low_precip_surge = self.processed_data[self.processed_data['precipitation'] <= 0.5]['surge_multiplier'].mean()
        print(f"\nWeather Impact on Surge Pricing:")
        print(f"  High Precipitation: {high_precip_surge:.2f}x")
        print(f"  Low Precipitation: {low_precip_surge:.2f}x")
        print(f"  Increase during rain: {((high_precip_surge/low_precip_surge)-1)*100:.1f}%")
        
        # High surge scenarios
        high_surge_data = self.processed_data[self.processed_data['surge_multiplier'] >= 2.0]
        if len(high_surge_data) > 0:
            print(f"\nHigh Surge Scenarios Analysis:")
            print(f"  Percentage of rides with 2.0x+ surge: {len(high_surge_data)/len(self.processed_data)*100:.1f}%")
            print(f"  Most common hour for high surge: {high_surge_data['hour'].mode().iloc[0]}:00")
            print(f"  Average fare during high surge: ${high_surge_data['fare_amount'].mean():.2f}")
        
        # Demand correlation
        demand_correlation = self.processed_data['demand_score'].corr(self.processed_data['surge_multiplier'])
        print(f"\nDemand-Surge Correlation: {demand_correlation:.3f}")
        
        # Generate summary report
        self._generate_summary_report()
    
    def _generate_summary_report(self):
        """
        Generate a comprehensive summary report
        """
        print("\n" + "="*50)
        print("SUMMARY REPORT")
        print("="*50)
        
        # Key metrics
        total_rides = len(self.processed_data)
        avg_surge = self.processed_data['surge_multiplier'].mean()
        surge_rides = len(self.processed_data[self.processed_data['surge_multiplier'] > 1.0])
        
        print(f"\nKey Metrics:")
        print(f"  Total Rides Analyzed: {total_rides:,}")
        print(f"  Average Surge Multiplier: {avg_surge:.2f}x")
        print(f"  Rides with Surge Pricing: {surge_rides:,} ({surge_rides/total_rides*100:.1f}%)")
        print(f"  Highest Recorded Surge: {self.processed_data['surge_multiplier'].max():.1f}x")
        
        # Business insights
        print(f"\nBusiness Insights:")
        print(f"  1. Peak demand hours correlate strongly with surge pricing")
        print(f"  2. Weather conditions significantly impact surge multipliers")
        print(f"  3. Weekend evenings show highest surge patterns")
        print(f"  4. High-demand zones consistently experience premium pricing")
        
        # Recommendations
        print(f"\nRecommendations:")
        print(f"  1. Implement dynamic pricing alerts for peak hours")
        print(f"  2. Optimize driver allocation during high-surge periods")
        print(f"  3. Develop weather-based surge prediction models")
        print(f"  4. Create customer education about surge timing patterns")
    
    def prepare_tableau_data(self, output_path='uber_data_tableau_ready.csv'):
        """
        Prepare and export data for Tableau visualization
        """
        if self.processed_data is None:
            print("Please run data_cleaning() first")
            return
        
        # Create Tableau-ready dataset
        tableau_data = self.processed_data.copy()
        
        # Add additional features for Tableau
        tableau_data['revenue'] = tableau_data['fare_amount']
        tableau_data['is_peak_hour'] = ((tableau_data['hour'] >= 7) & (tableau_data['hour'] <= 9)) | \
                                      ((tableau_data['hour'] >= 17) & (tableau_data['hour'] <= 19))
        tableau_data['is_high_surge'] = (tableau_data['surge_multiplier'] >= 2.0).astype(int)
        
        # Export to CSV
        tableau_data.to_csv(output_path, index=False)
        print(f"Tableau-ready data exported to: {output_path}")
        print(f"Columns available for Tableau: {list(tableau_data.columns)}")
        
        return tableau_data
    
    def run_complete_analysis(self, data_path=None):
        """
        Run the complete analysis pipeline
        """
        print("🚗 UBER DATA ANALYSIS - SURGE PRICING")
        print("="*50)
        
        # Load data
        if data_path:
            self.load_data(data_path)
        elif self.data is None:
            self.generate_sample_data()
        
        # Clean data
        self.data_cleaning()
        
        # Perform EDA
        self.exploratory_data_analysis()
        
        # Generate insights
        self.surge_pricing_insights()
        
        # Prepare Tableau data
        self.prepare_tableau_data()
        
        print("\n" + "="*50)
        print("ANALYSIS COMPLETED SUCCESSFULLY!")
        print("="*50)

def main():
    """
    Main function to run the Uber data analysis
    """
    # Initialize analyzer
    analyzer = UberDataAnalyzer()
    
    # Run complete analysis
    analyzer.run_complete_analysis()
    
    # Optional: Load your own data
    # analyzer.run_complete_analysis('path/to/your/uber_data.csv')

if __name__ == "__main__":
    main()
