import subprocess
import sys
from pathlib import Path

def run_script(script_name):
    print(f"Running {script_name}...")
    result = subprocess.run(
        ["python", f"scripts/{script_name}"],
        check=False,
        capture_output=True,
        text=True
    )
    
    if result.returncode != 0:
        print(f"Error in {script_name}:")
        print(result.stderr)
        sys.exit(1)
    
    print(result.stdout)
    return True

def main():
    # Ensure data directory exists
    Path('/app/data').mkdir(parents=True, exist_ok=True)
    
    # Run scripts in sequence
    scripts = [
        'truckerpathscraper.py',
        'reviewanalysis.py',
        'featurecheck.py'
    ]
    
    for script in scripts:
        if not run_script(script):
            sys.exit(1)
    
    print("All steps completed successfully.")
    
    # List generated files
    print("\nGenerated files:")
    for file in Path('/app/data').glob('*'):
        print(f"- {file.name}")

if __name__ == "__main__":
    main()