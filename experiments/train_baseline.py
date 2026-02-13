"""Train baseline XGBoost model."""
import sys
sys.path.append('.')
from src.models.baseline import BaselinePolicy

def main():
    print("Training baseline model...")
    policy = BaselinePolicy()
    # Training logic here
    print("Training complete!")

if __name__ == "__main__":
    main()
