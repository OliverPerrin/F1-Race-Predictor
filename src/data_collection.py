"""
F1 Race Predictor - Data Collection Module

This module fetches Formula 1 data from the FastF1 API.
Collects race results, qualifying data, and championship standings for the 2024-2025 seasons.

Author: Oliver Perrin
Date: 2025-11-03
"""
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import fastf1
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime
import time
import warnings

# Suppress FastF1 warnings for cleaner output
warnings.filterwarnings('ignore')

# Enable FastF1 cache for faster subsequent loads
cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.fastf1_cache')
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

# Create data directories
RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw')
os.makedirs(RAW_DATA_DIR, exist_ok=True)

class F1DataCollector:
    """Collects F1 data from FastF1."""
    
    def __init__(self, years=[2024, 2025]):
        """
        Initialize the data collector.
        Args:
            years (list): List of years to collect data for
        """
        self.years = years
        retry_config = Retry(
            total=3,
            connect=3,
            read=3,
            backoff_factor=1.0,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
        self.http = requests.Session()
        adapter = HTTPAdapter(max_retries=retry_config)
        self.http.mount("http://", adapter)
        self.http.mount("https://", adapter)
        self.http.headers.update({"User-Agent": "F1RacePredictor/1.0"})
        
    def collect_race_results(self):
        """
        Collect race results for all races in specified years.
        
        Returns:
            pd.DataFrame: DataFrame containing race results
        """
        print("Collecting race results...")
        all_results = []
        now_utc = pd.Timestamp.now(tz="UTC")

        for year in self.years:
            print(f"\n Processing {year} season...")
            # Get the event schedule for the year

            schedule = fastf1.get_event_schedule(year)
            # Filter for race events only (excluding testing)
            races = schedule[schedule['EventFormat'] != 'testing']
            
            for index, race in races.iterrows():
                race_name = race['EventName']
                round_number = race['RoundNumber']
                race_date = race.get('Session4Date', race.get('EventDate'))
                if pd.notna(race_date):
                    race_date = pd.to_datetime(race_date)
                    if race_date.tzinfo is None:
                        race_date = race_date.tz_localize('UTC')
                    else:
                        race_date = race_date.tz_convert('UTC')
                    if race_date > now_utc + pd.Timedelta(days=1):
                        print(f" Skipping {race_name} (scheduled for {race_date.date()})")
                        continue
                print(f" Loading {race_name} (Round {round_number})...")
                
                session = fastf1.get_session(year, int(round_number), 'R')
                session.load()
                
                results = session.results
                if results is None or results.empty:
                    print(f" No race results data for {race_name}; skipping")
                    continue
                results['Year'] = year
                results['RoundNumber'] = round_number
                results['RaceName'] = race_name
                results['Country'] = race['Country']
                results['Location'] = race['Location']
                results['EventDate'] = race['EventDate']
                
                all_results.append(results)
                print(f"{race_name} collected successfully")
                time.sleep(1)
        
        if all_results:
            df = pd.concat(all_results, ignore_index=True)
            print(f"\n Collected {len(df)} race results from {len(all_results)} races")
            return df
        else:
            print("\n No race results collected")
            return pd.DataFrame()
        
        
    def collect_qualifying_results(self):
        """
        Collect qualifying results for all races in specified years.
        Returns:
            pd.DataFrame: DataFrame containing qualifying results
        """
        print("\nCollecting qualifying results...")
        
        all_qualifying = []
        now_utc = pd.Timestamp.now(tz="UTC")
        
        for year in self.years:
            print(f"\nProcessing {year} season...")
            
            schedule = fastf1.get_event_schedule(year)
            races = schedule[schedule['EventFormat'] != 'testing']
            
            for index, event in races.iterrows():
                race_name = event['EventName']
                round_number = event['RoundNumber']
                quali_date = event.get('Session3Date', event.get('EventDate'))
                if pd.notna(quali_date):
                    quali_date = pd.to_datetime(quali_date)
                    if quali_date.tzinfo is None:
                        quali_date = quali_date.tz_localize('UTC')
                    else:
                        quali_date = quali_date.tz_convert('UTC')
                    if quali_date > now_utc + pd.Timedelta(days=1):
                        print(f" Skipping {race_name} qualifying (scheduled for {quali_date.date()})")
                        continue
                    
                print(f" Loading {race_name} (Round {round_number})...")
            
                session = fastf1.get_session(year, int(round_number), 'Q')
                session.load()
                    
                # Get qualifying results
                results = session.results
                if results is None or results.empty:
                    print(f" No qualifying results data for {race_name}; skipping")
                    continue
                        
                results['Year'] = year
                results['RoundNumber'] = round_number
                results['RaceName'] = race_name
                results['Country'] = event['Country']
                results['Location'] = event['Location']
                        
                all_qualifying.append(results)
                        
                print(f" {race_name} qualifying collected")
                        
                time.sleep(1)
                    
        if all_qualifying:
            df = pd.concat(all_qualifying, ignore_index=True)
            print(f"\nCollected {len(df)} qualifying results from {len(all_qualifying)} races")
            return df
        else:
            print("\nNo qualifying results collected")
            return pd.DataFrame()

    
    def save_data(self, race_df, quali_df):
        """
        Save collected data to CSV files. 
        Args:
            race_df: DataFrame with race results
            quali_df: DataFrame with qualifying results
        """
        print("\nSaving data to files...")
        
        
        if not race_df.empty:
            race_path = os.path.join(RAW_DATA_DIR, 'race_results.csv')
            race_df.to_csv(race_path, index=False)
            print(f" Race results saved to {race_path}")
            
        if not quali_df.empty:
            quali_path = os.path.join(RAW_DATA_DIR, 'qualifying_results.csv')
            quali_df.to_csv(quali_path, index=False)
            print(f" Qualifying results saved to {quali_path}")
            
            metadata = {
                'collection_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'years': self.years,
                'races_collected': len(race_df) if not race_df.empty else 0,
                'qualifying_collected': len(quali_df) if not quali_df.empty else 0,
            }
            
            metadata_path = os.path.join(RAW_DATA_DIR, 'metadata.json')
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            print(f" Metadata saved to {metadata_path}")
            
            print("\nAll data saved successfully!")


def main():
    print("=" * 60)
    print("🏎️  F1 Race Predictor - Data Collection")
    print("=" * 60)
    print(f"Author: Oliver Perrin")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Initialize collector
    collector = F1DataCollector(years=[2024, 2025])
    
    # Collect all data
    race_results = collector.collect_race_results()
    qualifying_results = collector.collect_qualifying_results()
    
    # Save data
    collector.save_data(
        race_results,
        qualifying_results
    )
    
    print("\n" + "=" * 60)
    print("Data collection complete!")
    print("=" * 60)
    
    # Print summary
    if not race_results.empty:
        print(f"\nSummary:")
        print(f" • Total races: {race_results['RaceName'].nunique()}")
        print(f" • Total race results: {len(race_results)}")
        print(f" • Unique drivers: {race_results['Abbreviation'].nunique()}")
        print(f" • Unique constructors: {race_results['TeamName'].nunique()}")
        print(f" • Years covered: {', '.join(map(str, sorted(race_results['Year'].unique())))}")


if __name__ == "__main__":
    main()
