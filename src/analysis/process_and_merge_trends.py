"""
Process and merge Google Trends data with master cliodynamics dataset.

This module smooths mobilization index data and merges it with the main dataset,
creating a weighted PSI index that incorporates public mobilization signals.
"""

import pandas as pd
from pathlib import Path


# Project root directory (two levels up from this script)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"


def load_and_validate_csv(filepath: Path, date_column: str = "date") -> pd.DataFrame:
    """
    Load CSV file and convert date column to datetime.
    
    Args:
        filepath: Path to the CSV file.
        date_column: Name of the column containing dates.
        
    Returns:
        DataFrame with parsed date column.
        
    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    df = pd.read_csv(filepath)
    df[date_column] = pd.to_datetime(df[date_column])
    return df


def smooth_mobilization_index(trends_df: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """
    Apply rolling mean smoothing to the mobilization index.
    
    Args:
        trends_df: DataFrame containing mobilization_index column.
        window: Rolling window size for smoothing.
        
    Returns:
        DataFrame with additional mobilization_smooth column.
    """
    trends_df = trends_df.copy()
    trends_df["mobilization_smooth"] = trends_df["mobilization_index"].rolling(
        window=window,
        center=True,
        min_periods=1
    ).mean()
    return trends_df


def calculate_weighted_psi(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate weighted PSI index using mobilization data.
    
    The formula: psi_weighted = psi_index * (1 + mobilization_smooth / 100)
    
    Args:
        df: DataFrame containing psi_index and mobilization_smooth columns.
        
    Returns:
        DataFrame with additional psi_weighted column if psi_index exists.
    """
    if "psi_index" in df.columns:
        df["psi_weighted"] = df["psi_index"] * (1 + df["mobilization_smooth"] / 100)
    return df


def process_and_merge_trends(
    master_path: Path,
    trends_path: Path,
    output_path: Path
) -> pd.DataFrame:
    """
    Main processing function: merge trends data with master dataset.
    
    Args:
        master_path: Path to master cliodynamics CSV file.
        trends_path: Path to Google Trends mobilization CSV file.
        output_path: Path for the output CSV file.
        
    Returns:
        Processed and merged DataFrame.
    """
    # Load data
    master_df = load_and_validate_csv(master_path)
    trends_df = load_and_validate_csv(trends_path)
    
    # Smooth mobilization index
    trends_df = smooth_mobilization_index(trends_df)
    
    # Merge datasets
    final_df = pd.merge(
        master_df,
        trends_df[["date", "mobilization_smooth"]],
        on="date",
        how="left"
    )
    
    # Interpolate missing values
    final_df["mobilization_smooth"] = final_df["mobilization_smooth"].interpolate(
        method="linear",
        limit_direction="both"
    )
    
    # Calculate weighted PSI
    final_df = calculate_weighted_psi(final_df)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save result
    final_df.to_csv(output_path, index=False)
    print(f"Output saved to: {output_path}")
    
    return final_df


def main():
    """Main entry point for the script."""
    master_file = DATA_PROCESSED / "master_cliodynamics_v2.csv"
    trends_file = DATA_RAW / "google_trends_mobilization.csv"
    output_file = DATA_PROCESSED / "master_cliodynamics_v3.csv"
    
    df = process_and_merge_trends(master_file, trends_file, output_file)
    
    # Display summary
    print("\nLast 10 rows of processed data:")
    display_columns = ["date", "wealth_pump", "mobilization_smooth", "psi_weighted"]
    available_columns = [col for col in display_columns if col in df.columns]
    print(df[available_columns].tail(10))


if __name__ == "__main__":
    main()