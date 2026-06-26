"""
NSE Trading Calendar for ML Regime Classifier
Fetches holidays dynamically from NSE API (2011+) with 2010 fallback
"""

from datetime import datetime, timedelta
from typing import List, Optional, Set, Dict
import pandas as pd
import requests
import json
from pathlib import Path


# Hardcoded 2010 holidays (API doesn't support years before 2011)
HOLIDAYS_2010 = [
    '2010-01-01',  # New Year
    '2010-01-26',  # Republic Day
    '2010-02-12',  # Mahashivratri
    '2010-03-01',  # Holi
    '2010-03-24',  # Ram Navmi
    '2010-04-02',  # Good Friday
    '2010-04-14',  # Dr. Babasaheb Ambedkar Jayanti
    '2010-09-10',  # Ramzan Id
    '2010-11-05',  # Diwali (Laxmi Puja)
    '2010-11-17',  # Bakri-Id
    '2010-12-17',  # Moharrum
]


class NSECalendar:
    """
    NSE (National Stock Exchange of India) trading calendar.
    Fetches holidays from NSE API for 2011+ with caching.
    """
    
    def __init__(self, cache_dir: str = 'data/ingestion/.cache'):
        """
        Initialize NSECalendar with caching.
        
        Args:
            cache_dir: Directory to cache API responses
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._holiday_cache: Dict[int, Set[str]] = {}
        
        # Preload 2010 holidays
        self._holiday_cache[2010] = set(HOLIDAYS_2010)
    
    def _get_cache_path(self, year: int) -> Path:
        """Get cache file path for a given year."""
        return self.cache_dir / f'holidays_{year}.json'
    
    def _fetch_holidays_from_api(self, year: int) -> Optional[Set[str]]:
        """
        Fetch holidays from NSE API for a given year.
        
        Args:
            year: Year to fetch holidays for (must be >= 2011)
            
        Returns:
            Set of holiday dates in 'YYYY-MM-DD' format, or None if fetch fails
        """
        if year < 2011:
            return None
        
        url = f'https://www.nseindia.com/api/holiday-master?type=trading&year={year}'
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Extract CM (Cash Market) holidays
            cm_holidays = data.get('CM', [])
            
            holidays = set()
            for holiday in cm_holidays:
                # Parse date format: '26-Jan-2011' -> '2011-01-26'
                date_str = holiday['tradingDate']
                date_obj = datetime.strptime(date_str, '%d-%b-%Y')
                holidays.add(date_obj.strftime('%Y-%m-%d'))
            
            return holidays
            
        except Exception as e:
            print(f"Warning: Failed to fetch holidays for {year} from NSE API: {e}")
            return None
    
    def _load_from_cache(self, year: int) -> Optional[Set[str]]:
        """Load holidays from cache file."""
        cache_path = self._get_cache_path(year)
        
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    data = json.load(f)
                return set(data['holidays'])
            except Exception as e:
                print(f"Warning: Failed to load cache for {year}: {e}")
                return None
        
        return None
    
    def _save_to_cache(self, year: int, holidays: Set[str]):
        """Save holidays to cache file."""
        cache_path = self._get_cache_path(year)
        
        try:
            with open(cache_path, 'w') as f:
                json.dump({
                    'year': year,
                    'holidays': sorted(list(holidays)),
                    'cached_at': datetime.now().isoformat()
                }, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save cache for {year}: {e}")
    
    def get_holidays(self, year: int) -> Set[str]:
        """
        Get all NSE trading holidays for a given year.
        
        Args:
            year: Year to get holidays for
            
        Returns:
            Set of holiday dates in 'YYYY-MM-DD' format
        """
        # Check memory cache
        if year in self._holiday_cache:
            return self._holiday_cache[year]
        
        # Check disk cache
        cached = self._load_from_cache(year)
        if cached is not None:
            self._holiday_cache[year] = cached
            return cached
        
        # Fetch from API (2011+)
        if year >= 2011:
            holidays = self._fetch_holidays_from_api(year)
            if holidays is not None:
                self._save_to_cache(year, holidays)
                self._holiday_cache[year] = holidays
                return holidays
        
        # Fallback: return empty set
        print(f"Warning: No holiday data available for {year}")
        return set()
    
    def is_trading_day(self, date: datetime) -> bool:
        """
        Check if a given date is a trading day.
        
        Args:
            date: Date to check
            
        Returns:
            True if trading day, False if weekend or holiday
        """
        # Check if weekend
        if date.weekday() >= 5:  # Saturday=5, Sunday=6
            return False
        
        # Check if holiday
        date_str = date.strftime('%Y-%m-%d')
        year = date.year
        holidays = self.get_holidays(year)
        
        return date_str not in holidays
    
    def get_trading_days(self, start_date: datetime, end_date: datetime) -> List[datetime]:
        """
        Get all trading days between start and end dates (inclusive).
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            List of trading days
        """
        trading_days = []
        current = start_date
        
        while current <= end_date:
            if self.is_trading_day(current):
                trading_days.append(current)
            current += timedelta(days=1)
        
        return trading_days
    
    def next_trading_day(self, date: datetime) -> datetime:
        """
        Get the next trading day after the given date.
        
        Args:
            date: Reference date
            
        Returns:
            Next trading day
        """
        next_day = date + timedelta(days=1)
        
        while not self.is_trading_day(next_day):
            next_day += timedelta(days=1)
        
        return next_day
    
    def previous_trading_day(self, date: datetime) -> datetime:
        """
        Get the previous trading day before the given date.
        
        Args:
            date: Reference date
            
        Returns:
            Previous trading day
        """
        prev_day = date - timedelta(days=1)
        
        while not self.is_trading_day(prev_day):
            prev_day -= timedelta(days=1)
        
        return prev_day


def validate_calendar_alignment(df: pd.DataFrame, date_col: str = 'Date') -> dict:
    """
    Validate that DataFrame dates align with NSE trading calendar.
    
    Args:
        df: DataFrame with date column
        date_col: Name of date column
        
    Returns:
        Dictionary with validation results
    """
    cal = NSECalendar()
    
    df[date_col] = pd.to_datetime(df[date_col])
    
    start_date = df[date_col].min()
    end_date = df[date_col].max()
    
    # Get expected trading days
    expected_days = cal.get_trading_days(start_date, end_date)
    actual_days = set(df[date_col].dt.date)
    expected_days_set = set([d.date() for d in expected_days])
    
    # Find mismatches
    missing_days = expected_days_set - actual_days
    extra_days = actual_days - expected_days_set
    
    return {
        'start_date': start_date,
        'end_date': end_date,
        'expected_trading_days': len(expected_days),
        'actual_days': len(actual_days),
        'missing_days': sorted(list(missing_days)),
        'extra_days': sorted(list(extra_days)),
        'coverage_pct': 100 * len(actual_days) / len(expected_days) if expected_days else 0
    }


if __name__ == "__main__":
    # Test the calendar
    print("="*60)
    print("NSE Calendar Test - Dynamic API Fetch")
    print("="*60)
    
    cal = NSECalendar()
    
    # Test 2010 (hardcoded)
    print("\n2010 Holidays (hardcoded):")
    holidays_2010 = cal.get_holidays(2010)
    print(f"  Count: {len(holidays_2010)}")
    print(f"  Sample: {sorted(list(holidays_2010))[:5]}")
    
    # Test 2011 (API)
    print("\n2011 Holidays (from NSE API):")
    holidays_2011 = cal.get_holidays(2011)
    print(f"  Count: {len(holidays_2011)}")
    print(f"  Sample: {sorted(list(holidays_2011))[:5]}")
    
    # Test 2024 (API)
    print("\n2024 Holidays (from NSE API):")
    holidays_2024 = cal.get_holidays(2024)
    print(f"  Count: {len(holidays_2024)}")
    print(f"  Sample: {sorted(list(holidays_2024))[:5]}")
    
    # Test is_trading_day
    test_date = datetime(2024, 1, 26)  # Republic Day
    print(f"\nIs {test_date.date()} a trading day? {cal.is_trading_day(test_date)}")
    
    test_date2 = datetime(2024, 1, 25)  # Regular day
    print(f"Is {test_date2.date()} a trading day? {cal.is_trading_day(test_date2)}")
    
    # Test date range
    print("\nTrading days in Jan 2024:")
    trading_days = cal.get_trading_days(datetime(2024, 1, 1), datetime(2024, 1, 31))
    print(f"  Count: {len(trading_days)}")
    print(f"  Expected: ~20-21 (excluding weekends and holidays)")
    
    print("\n" + "="*60)
    print("✓ Calendar test complete")
    print("="*60)
