import pandas as pd
import numpy as np
import sys
import glob

# Known Historical NIFTY 50 values (Approximate Closes)
# 29 Jan 2010 - NIFTY 50 Close was ~4882
# 04 Jan 2010 - NIFTY 50 Close was ~5232

def verify_nifty_data(file_path):
    print(f"Reading {file_path}")
    df = pd.read_csv(file_path)
    
    # Clean up column spaces
    df.columns = df.columns.str.strip()
    
    # Parse dates
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Check 2010 values to differentiate between Nifty 50 and Nifty 500
    # NIFTY 50 on Jan 4, 2010 was around 5232
    # NIFTY 500 on Jan 4, 2010 was around 4323
    
    df_jan_2010 = df[(df['Date'].dt.year == 2010) & (df['Date'].dt.month == 1)]
    
    # Check multiple dates across the timeline
    dates_to_check = ['2010-01-04', '2015-01-01', '2020-01-01', '2024-01-01', '2026-03-19']
    
    print(f"\n--- Checking specific longitudinal dates for {file_path} ---")
    for check_date in dates_to_check:
        row = df[df['Date'] == check_date]
        if not row.empty:
            close_val = row['Close'].values[0]
            print(f"Date: {check_date} | Close Value: {close_val}")
        else:
            # Try to get the closest available date if the exact one is a holiday
            closest_rows = df[df['Date'] >= check_date].sort_values('Date')
            if not closest_rows.empty:
                closest_date = closest_rows['Date'].dt.strftime('%Y-%m-%d').values[0]
                close_val = closest_rows['Close'].values[0]
                print(f"Date: {check_date} (Holiday) -> Next available: {closest_date} | Close Value: {close_val}")

    print("\n--- Recent 5 Rows ---")
    print(df.sort_values('Date', ascending=False).head(5)[['Date', 'Close', 'Volume']].to_string(index=False))
        
    print("\nTotal Rows Checked:", len(df))
    print("Earliest Date:", df['Date'].min())
    print("Latest Date:", df['Date'].max())

if __name__ == "__main__":
    nifty_files = glob.glob("../NIFTY*.csv")
    for nf in nifty_files:
        verify_nifty_data(nf)
