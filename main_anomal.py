import argparse
from DeepExperimentManager import ExperimentManager
from Module import AnomalTester

def main():
    """
    Main entry point for running the experiment.

    This script will:
    - Parse command-line arguments.
    - Initialize and run the ExperimentManager with the given config file.
    """
    parser = argparse.ArgumentParser(description="Run experiment using ExperimentManager.")
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file.')
    args = parser.parse_args()

    # Initialize ExperimentManager with the given config
    manager = ExperimentManager(config_path=args.config)
    manager.tester = AnomalTester # Change Tester Module for Anomaly Detection
    # Run the full experiment (training and optional testing)
    manager.run_experiment()

if __name__ == '__main__':
    main()
