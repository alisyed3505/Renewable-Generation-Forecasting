import requests
import json

API_KEY = 'yzC0UuVu9DZakpdfen7juaRMGHc_S3VE'
# Coordinates for a location in Germany (e.g., Berlin) to match our training context
LAT = 52.5200
LON = 13.4050

def test_solcast():
    print("Testing Solcast API...")
    
    # Try to get "Estimated Actuals" (Recent History) - This is what LSTM needs
    # Using the 'live' radiation endpoint if available, or world_radiation
    # Note: The exact endpoint depends on the plan. Let's try the generic world radiation one.
    
    url = f"https://api.solcast.com.au/world_radiation/estimated_actuals?latitude={LAT}&longitude={LON}&hours=24&format=json&api_key={API_KEY}"
    
    try:
        response = requests.get(url)
        print(f"Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print("Success! Data received.")
            if 'estimated_actuals' in data:
                print(f"Received {len(data['estimated_actuals'])} data points.")
                print("Sample data point:")
                print(json.dumps(data['estimated_actuals'][0], indent=2))
                return True
            else:
                print("Response format unexpected:", data.keys())
        else:
            print(f"Error: {response.text}")
            
            # If that failed, user might need to create a 'Resource' in the dashboard
            print("\nTip: If this failed with 404 or 403, you might need to create a 'Site' in the Solcast dashboard and use the Resource ID.")
            
    except Exception as e:
        print(f"Request failed: {e}")
        
    return False

if __name__ == "__main__":
    test_solcast()
