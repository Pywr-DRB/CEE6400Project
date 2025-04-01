import pandas as pd

def load_results(file_path: str) -> pd.DataFrame:
    """
    Load results from a CSV file and return a DataFrame.

    Args:
        file_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: DataFrame containing the results.
    """
    
    # Load the results from the CSV file
    results = pd.read_csv(file_path)
    
    return results