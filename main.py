import argparse
import sys
import config
from train_lstm import train_lstm
from inference_lstm import load_lstm_model, load_scaler, predict_realtime_lstm
from fetch_live_data import prepare_live_sequence

def main():
    parser = argparse.ArgumentParser(description="Solar Power Forecasting App")
    parser.add_argument('--train', action='store_true', help='Train the LSTM model')
    parser.add_argument('--predict', action='store_true', help='Run prediction with dummy data')
    parser.add_argument('--live', action='store_true', help='Run prediction with LIVE API data')
    
    args = parser.parse_args()
    
    if args.train:
        print("Starting training process...")
        train_lstm(
            data_path=config.DATA_FILE,
            model_path=config.MODEL_FILE,
            time_steps=config.TIME_STEPS
        )
        
    elif args.predict:
        print("Loading model for prediction...")
        model = load_lstm_model(config.MODEL_FILE)
        scaler = load_scaler(config.SCALER_FILE)
        
        if model and scaler:
            # Generate dummy data based on config features
            print("Generating dummy sequence for testing...")
            dummy_data = []
            for i in range(config.TIME_STEPS):
                # Simple dummy row
                row = {col: 0 for col in config.FEATURE_COLS}
                # Add some fake sun
                if 6 <= i <= 18:
                    row['SolarRadiationGlobalAt0'] = 500
                dummy_data.append(row)
                
            pred = predict_realtime_lstm(model, scaler, dummy_data, time_steps=config.TIME_STEPS)
            if pred is not None:
                print(f"\nPredicted Power Output: {pred:.4f} (Normalized)")
        else:
            print("Model or scaler not found. Please train first.")
            
    elif args.live:
        print("Loading model for LIVE prediction...")
        model = load_lstm_model(config.MODEL_FILE)
        scaler = load_scaler(config.SCALER_FILE)
        
        if model and scaler:
            print("Fetching live data from APIs...")
            live_sequence = prepare_live_sequence()
            
            if live_sequence:
                pred = predict_realtime_lstm(model, scaler, live_sequence, time_steps=config.TIME_STEPS)
                if pred is not None:
                    print(f"\n>>> LIVE FORECAST <<<")
                    print(f"Predicted Power Output for next hour: {pred:.4f} (Normalized)")
                    print("Note: Multiply this by your system's capacity (kW) to get actual power.")
            else:
                print("Failed to fetch live data.")
        else:
            print("Model or scaler not found. Please train first.")

    else:
        parser.print_help()

if __name__ == "__main__":
    main()
