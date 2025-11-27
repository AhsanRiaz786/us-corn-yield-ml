"""
Download county-level soil properties from USDA NRCS Soil Data Access API.

Retrieves aggregated soil properties for US counties using the gSSURGO database:
- Available Water Capacity (AWC)
- Clay content percentage
- Soil pH
- Organic matter percentage

Source: USDA Natural Resources Conservation Service - Soil Data Access
"""

import requests
import pandas as pd
import time
from pathlib import Path
from tqdm import tqdm

class SoilDataDownloader:
    """Download and aggregate county-level soil properties from USDA NRCS."""
    
    def __init__(self):
        self.base_url = "https://sdmdataaccess.sc.egov.usda.gov/tabular/post.rest"
        self.output_dir = Path(__file__).parent.parent.parent / "data" / "raw"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def query_soil_data(self, state_fips, county_fips):
        """
        Query soil properties for a specific county.
        
        Args:
            state_fips: State FIPS code (2 digits)
            county_fips: County FIPS code (3 digits)
            
        Returns:
            Dictionary of soil properties or None if query fails
        """
        areasymbol = f"US{state_fips}{county_fips}"
        
        query = f"""
        SELECT 
            co.areasymbol,
            AVG(awc.awc_r) as awc_avg,
            AVG(ch.claytotal_r) as clay_avg,
            AVG(ch.ph1to1h2o_r) as ph_avg,
            AVG(ch.om_r) as om_avg
        FROM legend AS l
        INNER JOIN mapunit AS mu ON mu.lkey = l.lkey
        INNER JOIN component AS co ON co.mukey = mu.mukey
        INNER JOIN chorizon AS ch ON ch.cokey = co.cokey
        LEFT JOIN chfrags AS cf ON cf.chkey = ch.chkey
        LEFT JOIN chtexturegrp AS chtg ON chtg.chkey = ch.chkey
        LEFT JOIN chtexture AS cht ON cht.chtgkey = chtg.chtgkey
        LEFT JOIN copmgrp AS cpm ON cpm.cokey = co.cokey
        LEFT JOIN corestrictions AS cr ON cr.cokey = co.cokey
        LEFT JOIN cosurfmorphgc AS cosmg ON cosmg.cokey = co.cokey
        LEFT JOIN cosurfmorphhpp AS cosmhpp ON cosmhpp.cokey = co.cokey
        LEFT JOIN cosurfmorphmr AS cosmr ON cosmr.cokey = co.cokey
        LEFT JOIN cosurfmorphss AS cosmss ON cosmss.cokey = co.cokey
        LEFT JOIN chstructgrp AS chsg ON chsg.chkey = ch.chkey
        LEFT JOIN chstruct AS chs ON chs.chstructgkey = chsg.chstructgkey
        LEFT JOIN chconsistence AS chc ON chc.chkey = ch.chkey
        LEFT JOIN cogeomordesc AS cgmd ON cgmd.cokey = co.cokey
        LEFT JOIN comonth AS cm ON cm.cokey = co.cokey
        LEFT JOIN cosoilmoist AS csm ON csm.cokey = co.cokey AND csm.comonthkey = cm.comonthkey
        LEFT JOIN cosoiltemp AS cst ON cst.cokey = co.cokey AND cst.comonthkey = cm.comonthkey
        LEFT JOIN chpores AS chp ON chp.chkey = ch.chkey
        LEFT JOIN chtext AS chtxt ON chtxt.chkey = ch.chkey
        LEFT JOIN chdesgnsuffix AS chds ON chds.chkey = ch.chkey
        LEFT JOIN chtexturemod AS chtm ON chtm.chtkey = cht.chtkey
        LEFT JOIN component AS awc ON awc.cokey = co.cokey
        WHERE l.areasymbol = '{areasymbol}'
        GROUP BY co.areasymbol
        """
        
        try:
            response = requests.post(
                self.base_url,
                data={"query": query, "format": "JSON"},
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'Table' in data and len(data['Table']) > 0:
                    row = data['Table'][0]
                    return {
                        'State_FIPS': state_fips,
                        'County_FIPS': county_fips,
                        'AWC_avg': row.get('awc_avg'),
                        'Clay_avg': row.get('clay_avg'),
                        'pH_avg': row.get('ph_avg'),
                        'OM_avg': row.get('om_avg')
                    }
            return None
            
        except Exception as e:
            return None
    
    def get_all_county_fips(self):
        """
        Load county FIPS codes from existing data files.
        
        Returns:
            DataFrame with State and County FIPS codes
        """
        yield_file = self.output_dir / "us_corn_yield_county_all_years.csv"
        
        if not yield_file.exists():
            raise FileNotFoundError(
                "Yield data not found. Please run get_yield_by_county.py first."
            )
        
        df = pd.read_csv(yield_file)
        
        counties = df[['State ANSI', 'County ANSI']].drop_duplicates()
        counties.columns = ['State_FIPS', 'County_FIPS']
        
        counties['State_FIPS'] = counties['State_FIPS'].astype(str).str.zfill(2)
        counties['County_FIPS'] = counties['County_FIPS'].astype(str).str.zfill(3)
        
        return counties.sort_values(['State_FIPS', 'County_FIPS']).reset_index(drop=True)
    
    def download_all_soil_data(self):
        """
        Download soil data for all counties with rate limiting.
        
        Returns:
            DataFrame with county soil properties
        """
        counties = self.get_all_county_fips()
        print(f"Found {len(counties)} counties to process")
        
        results = []
        
        for idx, row in tqdm(counties.iterrows(), total=len(counties), 
                            desc="Downloading soil data"):
            soil_data = self.query_soil_data(row['State_FIPS'], row['County_FIPS'])
            
            if soil_data:
                results.append(soil_data)
            
            if (idx + 1) % 50 == 0:
                time.sleep(2)
            else:
                time.sleep(0.5)
        
        df_soil = pd.DataFrame(results)
        
        output_file = self.output_dir / "county_soil_aggregates.csv"
        df_soil.to_csv(output_file, index=False)
        
        print(f"\nSoil data download complete!")
        print(f"Retrieved data for {len(results)} counties")
        print(f"Saved to: {output_file}")
        
        missing = len(counties) - len(results)
        if missing > 0:
            print(f"Warning: {missing} counties have no soil data available")
        
        print("\nSoil property summary:")
        print(df_soil[['AWC_avg', 'Clay_avg', 'pH_avg', 'OM_avg']].describe())
        
        return df_soil


def main():
    """Main execution function."""
    downloader = SoilDataDownloader()
    df_soil = downloader.download_all_soil_data()
    
    print("\nSample data:")
    print(df_soil.head(10))


if __name__ == "__main__":
    main()

