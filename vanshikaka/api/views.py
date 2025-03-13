from django.http import JsonResponse
from django.shortcuts import render, redirect
import json
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from datetime import datetime, timedelta
from sklearn.linear_model import LinearRegression
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods
from .chatbot import EcoMetricsChatbot

# Initialize the chatbot
chatbot = EcoMetricsChatbot()

def index(request):
    return render(request, 'application/index.html')

def dashboard1(request):
    return render(request, 'application/dashboard1.html')

def e_waste_chart(request):
    return render(request, 'application/dashboard2.html')

# Keep dashboard2 as an alias for e_waste_chart
def dashboard2(request):
    return render(request, 'application/dashboard2.html')

def get_json_data(request):
    # Define the correct JSON file path as a string
    json_file_path = r"D:\vanshika ka project\vanshikaka\api\e_waste_dataset_cleaned.json"

    # Read the JSON file
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return JsonResponse(data, safe=False)
    except FileNotFoundError:
        return JsonResponse({"error": "File not found"}, status=404)
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON format"}, status=400)

def get_e_waste_data(request):
    try:
        with open('api/e_waste_dataset_cleaned.json', 'r') as file:
            data = json.load(file)
        return JsonResponse(data, safe=False)
    except FileNotFoundError:
        return JsonResponse({'error': 'Dataset not found'}, status=404)

def forecast_e_waste(request):
    try:
        # Load and prepare data
        with open('api/e_waste_dataset_cleaned.json', 'r') as file:
            data = json.load(file)
        
        # Convert list to DataFrame
        df = pd.DataFrame(data)
        
        # Ensure required columns exist
        required_columns = ['year_of_purchase', 'brand', 'weight_kg', 'quantity', 'recycled_price_usd', 'carbon_footprint']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")
        
        # Convert columns to numeric
        df['year'] = pd.to_numeric(df['year_of_purchase'], errors='coerce')
        df['weight_kg'] = pd.to_numeric(df['weight_kg'], errors='coerce')
        df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
        df['recycled_price_usd'] = pd.to_numeric(df['recycled_price_usd'], errors='coerce')
        df['carbon_footprint'] = pd.to_numeric(df['carbon_footprint'], errors='coerce')
        
        # Calculate e-waste generated (weight * quantity)
        df['e_waste_generated'] = df['weight_kg'] * df['quantity']
        
        # Calculate recycled amount based on recycled price
        df['recycled_amount'] = df['e_waste_generated'] * (df['recycled_price_usd'] / 100)  # Assuming recycling rate based on price
        
        # Remove rows with invalid years
        df = df.dropna(subset=['year'])
        df = df.sort_values('year')
        
        forecasts = {
            'brand_forecast': get_brand_forecast(df),
            'recycling_forecast': get_recycling_forecast(df),
            'yearly_forecast': get_yearly_forecast(df),
            'environmental_impact': get_environmental_forecast(df)
        }
        
        return JsonResponse({
            'success': True,
            'forecasts': forecasts
        })
    except Exception as e:
        print(f"Error in forecast_e_waste: {str(e)}")  # Add debug print
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

def get_brand_forecast(df):
    try:
        # Group by brand and year
        brand_yearly = df.groupby(['brand', 'year'])['e_waste_generated'].sum().reset_index()
        
        forecasts = {}
        for brand in brand_yearly['brand'].unique():
            brand_data = brand_yearly[brand_yearly['brand'] == brand]
            
            # Prepare features and target
            X = brand_data[['year']]
            y = brand_data['e_waste_generated']
            
            if len(X) < 2:  # Need at least 2 points for forecasting
                continue
            
            # Train RandomForest model
            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            # Generate future years (5 years ahead)
            last_year = int(max(X['year']))
            future_years = np.array(range(last_year + 1, last_year + 6)).reshape(-1, 1)
            
            # Make predictions
            predictions = model.predict(future_years)
            
            forecasts[brand] = {
                'years': future_years.flatten().tolist(),
                'predictions': predictions.tolist(),
                'confidence_intervals': get_confidence_intervals(predictions)
            }
        
        return forecasts
    except Exception as e:
        print(f"Error in brand forecast: {str(e)}")
        return {}

def get_recycling_forecast(df):
    try:
        # Calculate recycling rates
        df['recycling_rate'] = (df['recycled_amount'] / df['e_waste_generated'] * 100).fillna(0)
        yearly_rates = df.groupby('year')['recycling_rate'].mean()
        
        if len(yearly_rates) < 4:  # Need at least 4 points for ARIMA
            # Use simple linear regression instead
            X = np.array(range(len(yearly_rates))).reshape(-1, 1)
            y = yearly_rates.values
            model = LinearRegression()
            model.fit(X, y)
            
            # Forecast next 5 years
            future_X = np.array(range(len(yearly_rates), len(yearly_rates) + 5)).reshape(-1, 1)
            forecast = model.predict(future_X)
        else:
            # Use ARIMA model
            model = ARIMA(yearly_rates, order=(1,1,1))
            results = model.fit()
            forecast = results.forecast(steps=5)
        
        future_years = list(range(int(max(df['year'])) + 1, int(max(df['year'])) + 6))
        
        return {
            'years': future_years,
            'predictions': [max(0, min(100, x)) for x in forecast.tolist()],  # Ensure rates are between 0-100%
            'confidence_intervals': get_confidence_intervals(forecast)
        }
    except Exception as e:
        print(f"Error in recycling forecast: {str(e)}")
        return {
            'years': [],
            'predictions': [],
            'confidence_intervals': {'lower': [], 'upper': []}
        }

def get_yearly_forecast(df):
    try:
        # Group by year
        yearly_total = df.groupby('year')['e_waste_generated'].sum()
        
        if len(yearly_total) < 4:  # Need at least 4 points for ARIMA
            # Use simple linear regression instead
            X = np.array(range(len(yearly_total))).reshape(-1, 1)
            y = yearly_total.values
            model = LinearRegression()
            model.fit(X, y)
            
            # Forecast next 5 years
            future_X = np.array(range(len(yearly_total), len(yearly_total) + 5)).reshape(-1, 1)
            forecast = model.predict(future_X)
        else:
            # Use ARIMA model
            model = ARIMA(yearly_total, order=(1,1,1))
            results = model.fit()
            forecast = results.forecast(steps=5)
        
        future_years = list(range(int(max(df['year'])) + 1, int(max(df['year'])) + 6))
        
        return {
            'years': future_years,
            'predictions': [max(0, x) for x in forecast.tolist()],  # Ensure non-negative values
            'confidence_intervals': get_confidence_intervals(forecast)
        }
    except Exception as e:
        print(f"Error in yearly forecast: {str(e)}")
        return {
            'years': [],
            'predictions': [],
            'confidence_intervals': {'lower': [], 'upper': []}
        }

def get_environmental_forecast(df):
    try:
        # Calculate environmental impact metrics
        df['environmental_impact'] = df['carbon_footprint'] * df['e_waste_generated']
        yearly_impact = df.groupby('year')['environmental_impact'].sum()
        
        if len(yearly_impact) < 4:  # Need at least 4 points for ARIMA
            # Use simple linear regression instead
            X = np.array(range(len(yearly_impact))).reshape(-1, 1)
            y = yearly_impact.values
            model = LinearRegression()
            model.fit(X, y)
            
            # Forecast next 5 years
            future_X = np.array(range(len(yearly_impact), len(yearly_impact) + 5)).reshape(-1, 1)
            forecast = model.predict(future_X)
        else:
            # Use ARIMA model
            model = ARIMA(yearly_impact, order=(1,1,1))
            results = model.fit()
            forecast = results.forecast(steps=5)
        
        future_years = list(range(int(max(df['year'])) + 1, int(max(df['year'])) + 6))
        
        return {
            'years': future_years,
            'predictions': [max(0, x) for x in forecast.tolist()],  # Ensure non-negative values
            'confidence_intervals': get_confidence_intervals(forecast)
        }
    except Exception as e:
        print(f"Error in environmental forecast: {str(e)}")
        return {
            'years': [],
            'predictions': [],
            'confidence_intervals': {'lower': [], 'upper': []}
        }

def get_confidence_intervals(predictions, confidence=0.95):
    try:
        predictions = np.array(predictions)
        std = np.std(predictions)
        margin = 1.96 * std  # 95% confidence interval
        return {
            'lower': [max(0, pred - margin) for pred in predictions],  # Ensure non-negative values
            'upper': [pred + margin for pred in predictions]
        }
    except Exception as e:
        print(f"Error in confidence intervals: {str(e)}")
        return {'lower': [], 'upper': []}

def get_co2_data(request):
    try:
        # Get the absolute path to the JSON file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(current_dir, 'co2_waste_cleaned.json')
        
        # Read the JSON file
        with open(json_path, 'r') as file:
            data = json.load(file)
        
            return JsonResponse(data, safe=False)
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

def process_co2_analytics(request):
    try:
        # Get the absolute path to the JSON file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_file_path = os.path.join(current_dir, 'co2 waste cleaned.json')
        print(f"Processing analytics from: {json_file_path}")
        
        # Read JSON file
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            print("JSON data loaded successfully")
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        print(f"DataFrame created with columns: {df.columns.tolist()}")
        
        # Basic data cleaning
        df['co2_emissions'] = pd.to_numeric(df['CO2 Emissions (Metric Tons)'].astype(str).str.replace(',', ''), errors='coerce')
        df['emission_reduction'] = pd.to_numeric(df['Emission Reduction Target (%)'], errors='coerce')
        df['recycling_rate'] = pd.to_numeric(df['Recycling Rate (%)'], errors='coerce')
        print("Numeric conversions completed")
        
        # Simple analytics calculations
        total_co2 = float(df['co2_emissions'].sum())
        avg_reduction = float(df['emission_reduction'].mean())
        avg_recycling = float(df['recycling_rate'].mean())
        print(f"Basic calculations: Total CO2={total_co2}, Avg Reduction={avg_reduction}, Avg Recycling={avg_recycling}")
        
        # Prepare response data
        response_data = {
            'success': True,
            'analytics': {
                'total_co2': {
                    'total': total_co2,
                    'change_percent': 0.0  # Simplified for now
                },
                'emissions_by_method': df.groupby('Sector')['co2_emissions'].sum().to_dict(),
                'facility_reduction': df.groupby('Country')['emission_reduction'].mean().to_dict(),
                'emissions_per_kg': float(df['co2_emissions'].mean()),
                'reduction_efficiency': {
                    'current': avg_recycling,
                    'forecast': avg_recycling * 1.1
                }
            },
            'visualizations': {
                'recycling_methods': {
                    'labels': df['Sector'].unique().tolist(),
                    'data': df.groupby('Sector')['co2_emissions'].sum().tolist()
                },
                'category_emissions': {
                    'labels': df['Energy Source'].unique().tolist(),
                    'data': df.groupby('Energy Source')['co2_emissions'].sum().tolist()
                },
                'yearly_trend': {
                    'labels': sorted(df['Year'].unique().tolist()),
                    'data': df.groupby('Year')['co2_emissions'].sum().tolist()
                },
                'weight_emissions_price': {
                    'weight': df['co2_emissions'].tolist(),
                    'emissions': df['co2_emissions'].tolist(),
                    'price': df['recycling_rate'].tolist()
                }
            }
        }
        
        # Add simple forecasting data
        years = sorted(df['Year'].unique())
        current_emissions = df.groupby('Year')['co2_emissions'].sum().tolist()
        
        # Simple linear projection for forecast
        forecast_years = list(range(2025, 2031))
        last_value = current_emissions[-1]
        avg_change = (current_emissions[-1] - current_emissions[0]) / len(current_emissions)
        forecast_values = [last_value + (avg_change * i) for i in range(1, 7)]
        
        response_data['forecasts'] = {
            'co2_trends': {
                'years': forecast_years,
                'forecast': forecast_values,
                'confidence_intervals': {
                    'upper': [v * 1.1 for v in forecast_values],
                    'lower': [v * 0.9 for v in forecast_values]
                }
            }
        }
        
        print("Response data prepared successfully")
        return JsonResponse(response_data)
        
    except FileNotFoundError:
        print(f"File not found at: {json_file_path}")
        return JsonResponse({
            'success': False,
            'error': 'CO2 dataset not found'
        }, status=404)
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': 'Invalid JSON format in CO2 dataset'
        }, status=400)
    except Exception as e:
        print(f"Unexpected error in process_co2_analytics: {str(e)}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

def get_recycling_methods_data(df):
    return {
        'labels': df['recycling_method'].unique().tolist(),
        'data': df.groupby('recycling_method')['co2_emissions'].sum().tolist()
    }

def get_category_emissions(df):
    return {
        'labels': df['category'].unique().tolist(),
        'data': df.groupby('category')['co2_emissions'].sum().tolist()
    }

def get_location_emissions(df):
    return df.groupby(['latitude', 'longitude'])['co2_emissions'].sum().to_dict()

def get_yearly_emissions(df):
    yearly = df.groupby('year')['co2_emissions'].sum()
    return {
        'labels': yearly.index.tolist(),
        'data': yearly.values.tolist()
    }

def get_bubble_chart_data(df):
    return {
        'weight': df['weight_kg'].tolist(),
        'emissions': df['co2_emissions'].tolist(),
        'price': df['recycled_price_usd'].tolist(),
        'categories': df['category'].tolist()
    }

def forecast_co2_trends(df):
    yearly_emissions = df.groupby('year')['co2_emissions'].sum()
    model = ARIMA(yearly_emissions, order=(1,1,1))
    results = model.fit()
    forecast = results.forecast(steps=6)  # Forecast until 2030
    return {
        'years': list(range(2025, 2031)),
        'forecast': forecast.tolist(),
        'confidence_intervals': get_confidence_intervals(forecast)
    }

def get_efficiency_correlation(df):
    X = df[['emission_reduction']]
    y = df['co2_emissions']
    model = LinearRegression()
    model.fit(X, y)
    return {
        'slope': float(model.coef_[0]),
        'intercept': float(model.intercept_),
        'x_values': X['emission_reduction'].tolist(),
        'y_values': y.tolist()
    }

def forecast_emissions_per_kg(df):
    df['emissions_per_kg'] = df['co2_emissions'] / df['weight_kg']
    yearly_emissions_per_kg = df.groupby('year')['emissions_per_kg'].mean()
    model = ARIMA(yearly_emissions_per_kg, order=(1,1,1))
    results = model.fit()
    forecast = results.forecast(steps=6)
    return {
        'years': list(range(2025, 2031)),
        'forecast': forecast.tolist(),
        'confidence_intervals': get_confidence_intervals(forecast)
    }

def forecast_location_emissions(df):
    location_emissions = df.groupby(['latitude', 'longitude', 'year'])['co2_emissions'].sum().reset_index()
    forecasts = {}
    for location in location_emissions[['latitude', 'longitude']].drop_duplicates().values:
        loc_data = location_emissions[
            (location_emissions['latitude'] == location[0]) & 
            (location_emissions['longitude'] == location[1])
        ]
        if len(loc_data) >= 4:  # Need at least 4 points for ARIMA
            model = ARIMA(loc_data['co2_emissions'], order=(1,1,1))
            results = model.fit()
            forecast = results.forecast(steps=6)
            forecasts[f"{location[0]},{location[1]}"] = forecast.tolist()
    return forecasts

def get_forecast_data(request):
    try:
        # Load and prepare data
        with open('api/e_waste_dataset_cleaned.json', 'r') as file:
            data = json.load(file)
        
        df = pd.DataFrame(data)
        
        # Calculate forecasts
        forecasts = {
            'e_waste': get_yearly_forecast(df),
            'recycling': get_recycling_forecast(df),
            'environmental': get_environmental_forecast(df)
        }
        
        return JsonResponse({
            'success': True,
            'forecasts': forecasts
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

def get_co2_kpis(request):
    try:
        # Get the absolute path to the JSON file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(current_dir, 'co2_waste_cleaned.json')
        
        # Read the JSON file
        with open(json_path, 'r') as file:
            data = json.load(file)
        
        # Calculate KPIs
        total_emissions = sum(float(item.get('CO2 Emissions (Metric Tons)', '0').replace(',', '')) for item in data)
        avg_intensity = sum(float(item.get('Emission Intensity', '0')) for item in data) / len(data)
        avg_recycling = sum(float(item.get('Recycling Rate (%)', '0')) for item in data) / len(data)
        
        # Calculate sector contributions
        sector_emissions = {}
        for item in data:
            sector = item.get('Sector', 'Unknown')
            emissions = float(item.get('CO2 Emissions (Metric Tons)', '0').replace(',', ''))
            sector_emissions[sector] = sector_emissions.get(sector, 0) + emissions
        
        return JsonResponse({
            'success': True,
            'kpis': {
                'total_emissions': {
                    'total': total_emissions,
                    'unit': 'MT'
                },
                'emission_intensity': {
                    'current': avg_intensity,
                    'unit': 'MT/USD'
                },
                'recycling_efficiency': {
                    'current': avg_recycling,
                    'unit': '%'
                },
                'sector_contribution': sector_emissions,
                'reduction_success': {
                    'rate': sum(float(item.get('Emission Reduction Target (%)', '0')) for item in data) / len(data),
                    'unit': '%'
                }
            }
        })
    except Exception as e:
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

def get_co2_visualizations(request):
    try:
        # Get the absolute path to the JSON file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        json_path = os.path.join(current_dir, 'co2_waste_cleaned.json')
        
        # Read the JSON file
        with open(json_path, 'r') as file:
            data = json.load(file)
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Clean numeric data
        df['co2_emissions'] = pd.to_numeric(df['CO2 Emissions (Metric Tons)'].astype(str).str.replace(',', ''), errors='coerce')
        df['emission_reduction'] = pd.to_numeric(df['Emission Reduction Target (%)'], errors='coerce')
        df['recycling_rate'] = pd.to_numeric(df['Recycling Rate (%)'], errors='coerce')
        df['year'] = pd.to_numeric(df['Year'], errors='coerce')
        df['global_pct'] = pd.to_numeric(df['Percentage of Global CO2 Emissions'], errors='coerce')
        
        # Remove rows with NaN values in critical columns
        df = df.dropna(subset=['co2_emissions', 'year'])
        print("Data cleaning completed")
        
        try:
            # 1. Bar Chart - Sector-wise CO2 emissions
            sector_emissions = df.groupby('Sector')['co2_emissions'].sum().sort_values(ascending=False)
            sector_data = {
                'labels': sector_emissions.index.tolist(),
                'data': [float(x) for x in sector_emissions.values.tolist()]
            }
            print("Sector data processed")
            
            # 2. Energy Source Contribution (Pie Chart)
            energy_source = df.groupby('Energy Source')['co2_emissions'].sum()
            energy_data = {
                'labels': energy_source.index.tolist(),
                'data': [float(x) for x in energy_source.values.tolist()]
            }
            print("Energy source data processed")
            
            # 3. Yearly trend data
            yearly_data = df.groupby('Year')['co2_emissions'].sum().sort_index()
            trend_data = {
                'labels': [str(x) for x in yearly_data.index.tolist()],
                'data': [float(x) for x in yearly_data.values.tolist()]
            }
            print("Yearly trend data processed")
            
            # 4. Bubble Chart - Recycling vs Emissions
            bubble_data = {
                'x': df['co2_emissions'].fillna(0).tolist(),
                'y': df['recycling_rate'].fillna(0).tolist(),
                'size': df['global_pct'].fillna(1).tolist(),
                'categories': df['Sector'].fillna('Unknown').tolist()
            }
            print("Bubble chart data processed")
            
            # 5. Geographic data
            geo_data = df.groupby('Country')['co2_emissions'].sum().to_dict()
            geo_data = {k: float(v) for k, v in geo_data.items()}
            print("Geographic data processed")
            
            # Forecasting Data
            # 1. ARIMA Forecast
            yearly_emissions = df.groupby('Year')['co2_emissions'].sum()
            
            if len(yearly_emissions) >= 4:
                try:
                    # Convert index to numeric if it's not already
                    yearly_emissions.index = pd.to_numeric(yearly_emissions.index)
                    yearly_emissions = yearly_emissions.sort_index()
                    
                    model = ARIMA(yearly_emissions, order=(1,1,1))
                    results = model.fit()
                    forecast = results.forecast(steps=6)
                    forecast_data = {
                        'years': list(range(2025, 2031)),
                        'values': [float(x) for x in forecast],
                        'confidence': {
                            'upper': [float(x * 1.1) for x in forecast],
                            'lower': [float(x * 0.9) for x in forecast]
                        }
                    }
                except Exception as e:
                    print(f"ARIMA failed, using linear regression: {str(e)}")
                    # Fallback to simple linear regression
                    X = np.array(range(len(yearly_emissions))).reshape(-1, 1)
                    y = yearly_emissions.values
                    model = LinearRegression()
                    model.fit(X, y)
                    future_X = np.array(range(len(yearly_emissions), len(yearly_emissions) + 6)).reshape(-1, 1)
                    forecast = model.predict(future_X)
                    forecast_data = {
                        'years': list(range(2025, 2031)),
                        'values': [float(x) for x in forecast],
                        'confidence': {
                            'upper': [float(x * 1.1) for x in forecast],
                            'lower': [float(x * 0.9) for x in forecast]
                        }
                    }
            else:
                forecast_data = {
                    'years': list(range(2025, 2031)),
                    'values': [0] * 6,
                    'confidence': {'upper': [0] * 6, 'lower': [0] * 6}
                }
            print("Forecast data processed")
            
            # 2. Random Forest for Sector-wise Predictions
            sector_predictions = {}
            for sector in df['Sector'].unique():
                sector_df = df[df['Sector'] == sector].copy()
                if len(sector_df) >= 4:
                    try:
                        X = sector_df[['year', 'recycling_rate', 'global_pct']].fillna(0)
                        y = sector_df['co2_emissions'].fillna(0)
                        
                        model = RandomForestRegressor(n_estimators=100, random_state=42)
                        model.fit(X, y)
                        
                        future_years = pd.DataFrame({
                            'year': range(2025, 2031),
                            'recycling_rate': [sector_df['recycling_rate'].mean()] * 6,
                            'global_pct': [sector_df['global_pct'].mean()] * 6
                        })
                        
                        predictions = model.predict(future_years)
                        sector_predictions[sector] = [float(x) for x in predictions]
                    except Exception as e:
                        print(f"Error in Random Forest for sector {sector}: {str(e)}")
                        continue
            print("Sector predictions processed")
            
            response_data = {
                'success': True,
                'visualizations': {
                    'sector_emissions': sector_data,
                    'energy_sources': energy_data,
                    'yearly_trend': trend_data,
                    'recycling_emissions': bubble_data,
                    'geo_data': geo_data
                },
                'forecasting': {
                    'time_series': forecast_data,
                    'sector_predictions': sector_predictions
                }
            }
            
            return JsonResponse(response_data)
            
        except Exception as e:
            print(f"Error in data processing: {str(e)}")
            return JsonResponse({
                'success': False,
                'error': f"Error in data processing: {str(e)}"
            }, status=500)
            
    except Exception as e:
        print(f"Error in get_co2_visualizations: {str(e)}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return JsonResponse({
            'success': False,
            'error': str(e)
        }, status=500)

@csrf_exempt
@require_http_methods(["POST"])
def chat(request):
    try:
        data = json.loads(request.body)
        message = data.get('message', '')
        
        if not message:
            return JsonResponse({
                'success': False,
                'error': 'No message provided'
            }, status=400)

        # Get response from chatbot
        response = chatbot.get_response(message)
        
        return JsonResponse({
            'success': True,
            'response': response
        })
        
    except json.JSONDecodeError:
        return JsonResponse({
            'success': False,
            'error': 'Invalid JSON data'
        }, status=400)
    except Exception as e:
        print(f"Error in chat view: {str(e)}")
        return JsonResponse({
            'success': False,
            'error': 'Internal server error'
        }, status=500)

def strategies(request):
    try:
        # Get CO2 data
        current_dir = os.path.dirname(os.path.abspath(__file__))
        co2_file_path = os.path.join(current_dir, 'co2_waste_cleaned.json')
        with open(co2_file_path, 'r', encoding='utf-8') as file:
            co2_data = json.load(file)

        # Get E-waste data
        ewaste_file_path = os.path.join(current_dir, 'e_waste_dataset_cleaned.json')
        with open(ewaste_file_path, 'r', encoding='utf-8') as file:
            ewaste_data = json.load(file)

        # Process data for tables
        co2_df = pd.DataFrame(co2_data)
        ewaste_df = pd.DataFrame(ewaste_data)

        # Process CO2 data
        co2_df['CO2 Emissions (Metric Tons)'] = pd.to_numeric(co2_df['CO2 Emissions (Metric Tons)'].astype(str).str.replace(',', ''), errors='coerce')
        co2_strategies = co2_df.groupby(['Sector', 'Company Name']).agg({
            'CO2 Emissions (Metric Tons)': 'sum',
            'Emission Reduction Target (%)': 'mean',
            'Recycling Rate (%)': 'mean'
        }).reset_index()

        # Process E-waste data
        ewaste_df['total_waste'] = ewaste_df['weight_kg'].astype(float) * ewaste_df['quantity'].astype(float)
        ewaste_strategies = ewaste_df.groupby(['brand', 'product_type']).agg({
            'total_waste': 'sum',
            'recycled_price_usd': 'mean',
            'carbon_footprint': 'sum'
        }).reset_index()

        context = {
            'co2_strategies': co2_strategies.to_dict('records'),
            'ewaste_strategies': ewaste_strategies.to_dict('records'),
            'top_strategies': [
                {
                    'title': 'Smart Energy Management',
                    'description': 'Implement AI-driven energy monitoring systems to optimize consumption patterns and reduce CO2 emissions.',
                    'impact': '25-30% reduction in energy-related emissions'
                },
                {
                    'title': 'Circular Supply Chain',
                    'description': 'Develop closed-loop supply chains that minimize waste and maximize resource recovery.',
                    'impact': '40% increase in resource efficiency'
                },
                {
                    'title': 'Green Technology Integration',
                    'description': 'Adopt renewable energy sources and energy-efficient technologies across operations.',
                    'impact': '50% reduction in carbon footprint'
                },
                {
                    'title': 'E-waste Recovery Programs',
                    'description': 'Implement comprehensive e-waste collection and recycling programs.',
                    'impact': '70% increase in e-waste recovery'
                },
                {
                    'title': 'Sustainable Product Design',
                    'description': 'Design products for longevity, repairability, and recyclability.',
                    'impact': '35% reduction in product-related waste'
                }
            ]
        }
        
        return render(request, 'application/strategies.html', context)
    except Exception as e:
        print(f"Error in strategies view: {str(e)}")
        return JsonResponse({'error': str(e)}, status=500)