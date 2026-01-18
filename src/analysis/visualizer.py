import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PROCESSED = PROJECT_ROOT / 'data' / 'processed'
OUTPUT = PROJECT_ROOT / 'output'

def create_visualization(file_path):
    # Load the data
    df = pd.read_csv(file_path)
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Set the style
    plt.style.use('bmh')
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot the lines
    ax.plot(df['date'], df['wage_nominal'], label='Nominal Wage', marker='o', markersize=3)
    ax.plot(df['date'], df['rent'], label='Rent', marker='s', markersize=3)
    ax.plot(df['date'], df['wage_real'], label='Real Wage', marker='^', markersize=3)
    ax.plot(df['date'], df['wealth_pump'], label='Wealth Pump', linestyle='--', color='purple')
    
    # Add title and labels
    ax.set_title('Cliodynamics Data Overview', fontsize=16)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Index / Value', fontsize=12)
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    ax.legend(loc='best')
    
    # Rotate date labels for better readability
    plt.xticks(rotation=45)
    
    # Tweak spacing to prevent clipping of tick-labels
    plt.tight_layout()
    
    # Save the plot
    output_file = OUTPUT / 'cliodynamics_plot.png'
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")
    
    # Show the plot
    # plt.show() # Commented out to be suitable for headless execution/running as a script

if __name__ == "__main__":
    create_visualization(DATA_PROCESSED / 'master_cliodynamics.csv')
