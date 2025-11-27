import requests
import pandas as pd
import time
from pathlib import Path
import json

class USDAQuickStats:
    def __init__(self):
        self.base_url = "https://quickstats.nass.usda.gov"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36',
            'Accept': 'application/json, text/javascript, */*; q=0.01',
            'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
            'X-Requested-With': 'XMLHttpRequest',
            'Origin': self.base_url,
            'Referer': f'{self.base_url}/'
        })
        
    def get_uuid(self, params):
        """Get UUID for the query parameters"""
        url = f"{self.base_url}/uuid/encode"
        
        # Define breadcrumb order
        breadcrumb_order = []
        if 'source_desc' in params:
            breadcrumb_order.append('source_desc')
        if 'sector_desc' in params:
            breadcrumb_order.append('sector_desc')
        if 'group_desc' in params:
            breadcrumb_order.append('group_desc')
        if 'commodity_desc' in params:
            breadcrumb_order.append('commodity_desc')
        if 'short_desc' in params:
            breadcrumb_order.append('short_desc')
        if 'agg_level_desc' in params:
            breadcrumb_order.append('agg_level_desc')
        if 'year' in params:
            breadcrumb_order.append('year')
        
        # Prepare form data with multiple breadcrumb entries
        form_data = []
        for key, value in params.items():
            form_data.append((key, value))
        
        # Add breadcrumb entries (multiple values for same key)
        for bc in breadcrumb_order:
            form_data.append(('breadcrumb', bc))
        
        try:
            response = self.session.post(url, data=form_data, timeout=30)
            response.raise_for_status()
            uuid = response.text.strip().strip('"')
            return uuid
        except Exception as e:
            print(f"Error getting UUID: {e}")
            print(f"Response: {response.text if 'response' in locals() else 'No response'}")
            return None
    
    def download_csv(self, uuid, year, output_dir="data"):
        """Download CSV file for a given UUID"""
        Path(output_dir).mkdir(exist_ok=True)
        
        csv_url = f"{self.base_url}/data/spreadsheet/{uuid}.csv"
        output_file = f"{output_dir}/corn_yield_{year}.csv"
        
        try:
            print(f"Downloading data for year {year}...")
            response = self.session.get(csv_url, timeout=60)
            response.raise_for_status()
            
            with open(output_file, 'wb') as f:
                f.write(response.content)
            
            print(f"✓ Saved: {output_file}")
            return output_file
        except Exception as e:
            print(f"✗ Error downloading {year}: {e}")
            return None
    
    def get_corn_yield_data(self, start_year=1866, end_year=2026):
        """
        Get corn yield data at county level for all years
        
        Selections:
        - Sector: CROPS
        - Group: FIELD CROPS
        - Commodity: CORN
        - Data Item: CORN, GRAIN - YIELD, MEASURED IN BU / ACRE
        - Geographic Level: COUNTY
        - Year: Individual years from start_year to end_year
        """
        
        downloaded_files = []
        
        # Define the query parameters (excluding year which will vary)
        base_params = {
            'source_desc': 'SURVEY',
            'sector_desc': 'CROPS',
            'group_desc': 'FIELD CROPS',
            'commodity_desc': 'CORN',
            'short_desc': 'CORN, GRAIN - YIELD, MEASURED IN BU / ACRE',
            'agg_level_desc': 'COUNTY',
        }
        
        # Years to iterate through (from the HTML, years range from 1850 to 2026)
        years = list(range(start_year, end_year + 1))
        
        total_years = len(years)
        
        for idx, year in enumerate(years, 1):
            print(f"\n[{idx}/{total_years}] Processing year {year}...")
            
            # Create params for this year
            params = base_params.copy()
            params['year'] = str(year)
            
            # Get UUID for this query
            uuid = self.get_uuid(params)
            
            if uuid:
                print(f"  UUID: {uuid}")
            
                # Download the CSV
                csv_file = self.download_csv(uuid, year)
                
                if csv_file:
                    downloaded_files.append(csv_file)
                
                # Be polite - add a small delay between requests
                time.sleep(1)
            else:
                print(f"  ✗ Failed to get UUID for year {year}")
        
        return downloaded_files
    
    def merge_csv_files(self, file_list, output_file="us_corn_yield_county_all_years.csv"):
        """Merge all downloaded CSV files into one"""
        if not file_list:
            print("No files to merge!")
            return None
        
        print(f"\n{'='*60}")
        print("Merging all CSV files...")
        print(f"{'='*60}")
        
        dfs = []
        for file in file_list:
            try:
                df = pd.read_csv(file)
                dfs.append(df)
                print(f"✓ Loaded {file}: {len(df)} rows")
            except Exception as e:
                print(f"✗ Error loading {file}: {e}")
        
        if dfs:
            merged_df = pd.concat(dfs, ignore_index=True)
            merged_df.to_csv(output_file, index=False)
            print(f"\n{'='*60}")
            print(f"✓ Merged file saved: {output_file}")
            print(f"  Total rows: {len(merged_df):,}")
            print(f"  Total files merged: {len(dfs)}")
            print(f"{'='*60}")
            
            # Display some statistics
            if 'Year' in merged_df.columns:
                print(f"\nYear range: {merged_df['Year'].min()} - {merged_df['Year'].max()}")
            if 'State' in merged_df.columns:
                print(f"Number of states: {merged_df['State'].nunique()}")
            if 'County' in merged_df.columns:
                print(f"Number of counties: {merged_df['County'].nunique()}")
            
            return output_file
        else:
            print("No data to merge!")
            return None


def main():
    """Main function to run the data collection"""
    print("="*60)
    print("USDA NASS QuickStats - Corn Yield Data Collector")
    print("="*60)
    print("\nQuery Parameters:")
    print("  - Program: SURVEY")
    print("  - Sector: CROPS")
    print("  - Group: FIELD CROPS")
    print("  - Commodity: CORN")
    print("  - Data Item: CORN, GRAIN - YIELD, MEASURED IN BU / ACRE")
    print("  - Geographic Level: COUNTY")
    print("  - Years: 1866-2026")
    print("="*60)
    
    # Create scraper instance
    scraper = USDAQuickStats()
    
    # Get data for all years (you can adjust the year range)
    # Starting from 1866 as that's a reasonable starting point for corn data
    downloaded_files = scraper.get_corn_yield_data(start_year=1866, end_year=2026)
    
    # Merge all files
    if downloaded_files:
        merged_file = scraper.merge_csv_files(downloaded_files)
        
        if merged_file:
            print(f"\n✓ Success! Complete dataset saved to: {merged_file}")
    else:
        print("\n✗ No files were downloaded.")


if __name__ == "__main__":
    main()
