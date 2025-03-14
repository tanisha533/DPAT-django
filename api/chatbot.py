import faiss
import json
import os
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd
import requests

class EcoMetricsChatbot:
    def __init__(self):
        self.model = None
        self.index = None
        self.data_texts = []
        self.ewaste_data = None
        self.co2_data = None
        self.initialize_model()

    def initialize_model(self):
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Load FAISS index
            index_path = os.path.join(current_dir, 'faiss_index.bin')
            self.index = faiss.read_index(index_path)
            
            # Load data texts
            texts_path = os.path.join(current_dir, 'data_texts.json')
            with open(texts_path, 'r', encoding='utf-8') as f:
                self.data_texts = json.load(f)
            
            # Load original datasets for calculations
            ewaste_path = os.path.join(current_dir, 'e_waste_dataset_cleaned.json')
            co2_path = os.path.join(current_dir, 'co2_waste_cleaned.json')
            
            with open(ewaste_path, 'r', encoding='utf-8') as file:
                self.ewaste_data = json.load(file)
            
            with open(co2_path, 'r', encoding='utf-8') as file:
                self.co2_data = json.load(file)
            
            # Load model
            self.model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            
            print("Model, index, and data texts loaded successfully!")

        except Exception as e:
            print(f"Error initializing: {str(e)}")
            self.model = None
            self.index = None
            self.data_texts = []
            self.ewaste_data = None
            self.co2_data = None

    def search_web(self, query):
        try:
            # Using a more specific search query format
            search_queries = [
                f"{query} site:unep.org",  # UN Environment Programme
                f"{query} site:itu.int",    # International Telecommunication Union
                f"{query} site:who.int",    # World Health Organization
                f"{query} site:worldbank.org" # World Bank
            ]
            
            for search_query in search_queries:
                url = f"https://api.duckduckgo.com/?q={search_query}&format=json"
                response = requests.get(url)
                data = response.json()
                
                if data.get("Abstract"):
                    return data["Abstract"]
                elif data.get("RelatedTopics") and len(data["RelatedTopics"]) > 0:
                    return data["RelatedTopics"][0].get("Text", "")
            
            # Fallback to general search if no results from specific sites
            url = f"https://api.duckduckgo.com/?q={query}&format=json"
            response = requests.get(url)
            data = response.json()
            
            if data.get("Abstract"):
                return data["Abstract"]
            elif data.get("RelatedTopics") and len(data["RelatedTopics"]) > 0:
                return data["RelatedTopics"][0].get("Text", "")
            
            # If no results found, return predefined global forecast data
            if "2025" in query.lower() and "e-waste" in query.lower():
                return ("According to global forecasts, e-waste generation is projected to reach 57.4 million metric tonnes (Mt) "
                       "by 2025. This represents a significant increase from previous years, driven by higher consumption of "
                       "electronic devices, shorter device lifespans, and limited repair options.")
                
            return None
        except Exception as e:
            print(f"Error in web search: {str(e)}")
            return None

    def get_recycling_strategies(self):
        try:
            df = pd.DataFrame(self.co2_data)
            # Get unique recycling methods and their impact on CO2 reduction
            df['CO2 Emissions (Metric Tons)'] = pd.to_numeric(df['CO2 Emissions (Metric Tons)'].astype(str).str.replace(',', ''), errors='coerce')
            df['Recycling Rate (%)'] = pd.to_numeric(df['Recycling Rate (%)'], errors='coerce')
            
            # Calculate average CO2 reduction for different recycling rates
            recycling_impact = df.groupby(pd.qcut(df['Recycling Rate (%)'], 4))['CO2 Emissions (Metric Tons)'].mean()
            
            response = "**Recycling Strategies and Their Impact on CO2 Emissions:**\n\n"
            for rate_range, emissions in recycling_impact.items():
                response += f"- Recycling Rate {rate_range}: Average CO2 Emissions {emissions:,.2f} Metric Tons\n"
            return response
        except Exception as e:
            print(f"Error calculating recycling strategies: {str(e)}")
            return None

    def calculate_total_ewaste(self):
        try:
            df = pd.DataFrame(self.ewaste_data)
            df['weight_kg'] = pd.to_numeric(df['weight_kg'], errors='coerce')
            df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
            total_ewaste = (df['weight_kg'] * df['quantity']).sum()
            return total_ewaste
        except Exception as e:
            print(f"Error calculating total e-waste: {str(e)}")
            return None

    def calculate_brand_ewaste(self, brand_name):
        try:
            df = pd.DataFrame(self.ewaste_data)
            df['weight_kg'] = pd.to_numeric(df['weight_kg'], errors='coerce')
            df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
            brand_data = df[df['brand'].str.lower() == brand_name.lower()]
            total_ewaste = (brand_data['weight_kg'] * brand_data['quantity']).sum()
            return total_ewaste
        except Exception as e:
            print(f"Error calculating brand e-waste: {str(e)}")
            return None

    def get_co2_emissions_by_year(self):
        try:
            df = pd.DataFrame(self.co2_data)
            # Convert CO2 emissions to numeric, handling commas and invalid values
            df['CO2 Emissions (Metric Tons)'] = pd.to_numeric(
                df['CO2 Emissions (Metric Tons)'].astype(str).str.replace(',', ''), 
                errors='coerce'
            )
            # Group by year and sum emissions
            yearly_emissions = df.groupby('Year')['CO2 Emissions (Metric Tons)'].sum().round(2)
            # Sort by year
            yearly_emissions = yearly_emissions.sort_index()
            return yearly_emissions.to_dict()
        except Exception as e:
            print(f"Error calculating CO2 emissions: {str(e)}")
            return None

    def get_ewaste_forecast(self):
        try:
            # Get web data for 2025 forecast with specific query
            query = "global e-waste statistics forecast 2025 UNEP ITU official projection"
            web_info = self.search_web(query)
            
            # Get dataset information
            df = pd.DataFrame(self.ewaste_data)
            df['weight_kg'] = pd.to_numeric(df['weight_kg'], errors='coerce')
            df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
            df['total_waste'] = df['weight_kg'] * df['quantity']
            
            response = "**E-waste Statistics and Forecast:**\n\n"
            
            # Add web-sourced global forecast
            if web_info:
                response += f"**Global Forecast (2025):**\n{web_info}\n\n"
            
            # Add dataset information
            total_waste = df['total_waste'].sum()
            response += f"**Current Dataset Statistics:**\nTotal e-waste recorded in our dataset: {total_waste:,.2f} kg\n\n"
            
            # Add comparison note
            response += ("*Note: The global forecast of 57.4 million metric tonnes represents worldwide projections for 2025, "
                      "while our dataset shows currently recorded e-waste data. The significant difference is because our dataset "
                      "contains only a sample of e-waste records, while the global forecast accounts for worldwide generation "
                      "across all regions and sectors.*")
            
            return response
        except Exception as e:
            print(f"Error in e-waste forecast: {str(e)}")
            return None

    def get_co2_forecast(self):
        try:
            # Get web data for 2025 CO2 forecast
            query = "global CO2 emissions forecast 2025 UNEP climate change"
            web_info = self.search_web(query)
            
            # Get dataset information
            df = pd.DataFrame(self.co2_data)
            df['CO2 Emissions (Metric Tons)'] = pd.to_numeric(df['CO2 Emissions (Metric Tons)'].astype(str).str.replace(',', ''), errors='coerce')
            
            response = "**CO2 Emissions Statistics and Forecast:**\n\n"
            
            # Add web-sourced global forecast
            if web_info:
                response += f"**Global Forecast (2025):**\n{web_info}\n\n"
            
            # Add sector-wise analysis
            sector_emissions = df.groupby('Sector')['CO2 Emissions (Metric Tons)'].sum().sort_values(ascending=False)
            response += "**Top Contributing Sectors:**\n"
            for sector, emissions in sector_emissions.head(5).items():
                response += f"- {sector}: {emissions:,.2f} Metric Tons\n"
            
            # Add total from dataset
            total_co2 = df['CO2 Emissions (Metric Tons)'].sum()
            response += f"\n**Current Dataset Total:**\nTotal CO2 emissions recorded: {total_co2:,.2f} Metric Tons\n"
            
            return response
        except Exception as e:
            print(f"Error in CO2 forecast: {str(e)}")
            return None

    def get_comprehensive_analysis(self):
        try:
            response = "**Comprehensive Environmental Analysis:**\n\n"
            
            # 1. E-waste Analysis
            df_ewaste = pd.DataFrame(self.ewaste_data)
            df_ewaste['weight_kg'] = pd.to_numeric(df_ewaste['weight_kg'], errors='coerce')
            df_ewaste['quantity'] = pd.to_numeric(df_ewaste['quantity'], errors='coerce')
            df_ewaste['total_waste'] = df_ewaste['weight_kg'] * df_ewaste['quantity']
            
            # Brand analysis
            brand_waste = df_ewaste.groupby('brand')['total_waste'].sum().sort_values(ascending=False)
            response += "**Top E-waste Contributing Brands:**\n"
            for brand, waste in brand_waste.head(5).items():
                response += f"- {brand}: {waste:,.2f} kg\n"
            
            # 2. CO2 Analysis
            df_co2 = pd.DataFrame(self.co2_data)
            df_co2['CO2 Emissions (Metric Tons)'] = pd.to_numeric(df_co2['CO2 Emissions (Metric Tons)'].astype(str).str.replace(',', ''), errors='coerce')
            
            # Sector analysis
            sector_emissions = df_co2.groupby('Sector')['CO2 Emissions (Metric Tons)'].sum().sort_values(ascending=False)
            response += "\n**Top CO2 Emitting Sectors:**\n"
            for sector, emissions in sector_emissions.head(5).items():
                response += f"- {sector}: {emissions:,.2f} Metric Tons\n"
            
            # 3. Recycling Strategies
            df_co2['Recycling Rate (%)'] = pd.to_numeric(df_co2['Recycling Rate (%)'], errors='coerce')
            recycling_impact = df_co2.groupby(pd.qcut(df_co2['Recycling Rate (%)'], 4))['CO2 Emissions (Metric Tons)'].mean()
            
            response += "\n**Impact of Recycling Rates on CO2 Emissions:**\n"
            for rate_range, emissions in recycling_impact.items():
                response += f"- {rate_range}: {emissions:,.2f} Metric Tons\n"
            
            return response
        except Exception as e:
            print(f"Error in comprehensive analysis: {str(e)}")
            return None

    def get_combined_sector_analysis(self):
        try:
            response = "**Sector-wise Environmental Impact Analysis:**\n\n"
            
            # Get CO2 sector data
            df_co2 = pd.DataFrame(self.co2_data)
            df_co2['CO2 Emissions (Metric Tons)'] = pd.to_numeric(df_co2['CO2 Emissions (Metric Tons)'].astype(str).str.replace(',', ''), errors='coerce')
            sector_emissions = df_co2.groupby('Sector')['CO2 Emissions (Metric Tons)'].sum().sort_values(ascending=False)
            
            # Get web data for sector impact
            web_info = self.search_web("most polluting industrial sectors CO2 emissions global impact")
            
            response += "**Top CO2 Emitting Sectors (Dataset Analysis):**\n"
            for sector, emissions in sector_emissions.head(5).items():
                response += f"- {sector}: {emissions:,.2f} Metric Tons\n"
            
            if web_info:
                response += f"\n**Global Sector Impact (Web Data):**\n{web_info}\n"
            
            return response
        except Exception as e:
            print(f"Error in sector analysis: {str(e)}")
            return None

    def get_combined_brand_analysis(self):
        try:
            response = "**Brand-wise E-waste Generation Analysis:**\n\n"
            
            # Get brand data from dataset
            df = pd.DataFrame(self.ewaste_data)
            df['weight_kg'] = pd.to_numeric(df['weight_kg'], errors='coerce')
            df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
            df['total_waste'] = df['weight_kg'] * df['quantity']
            brand_waste = df.groupby('brand')['total_waste'].sum().sort_values(ascending=False)
            
            # Get web data for brand sustainability
            web_info = self.search_web("top electronics brands e-waste generation environmental impact")
            
            response += "**Top E-waste Contributing Brands (Dataset Analysis):**\n"
            for brand, waste in brand_waste.head(5).items():
                response += f"- {brand}: {waste:,.2f} kg\n"
            
            if web_info:
                response += f"\n**Global Brand Impact (Web Data):**\n{web_info}\n"
            
            return response
        except Exception as e:
            print(f"Error in brand analysis: {str(e)}")
            return None

    def get_combined_recycling_strategies(self):
        try:
            response = "**Comprehensive Recycling Strategy Analysis:**\n\n"
            
            # Get recycling impact from dataset
            df = pd.DataFrame(self.co2_data)
            df['CO2 Emissions (Metric Tons)'] = pd.to_numeric(df['CO2 Emissions (Metric Tons)'].astype(str).str.replace(',', ''), errors='coerce')
            df['Recycling Rate (%)'] = pd.to_numeric(df['Recycling Rate (%)'], errors='coerce')
            recycling_impact = df.groupby(pd.qcut(df['Recycling Rate (%)'], 4))['CO2 Emissions (Metric Tons)'].mean()
            
            # Get web data for recycling strategies
            web_info = self.search_web("effective e-waste and CO2 reduction recycling strategies global best practices")
            
            response += "**Impact of Recycling Rates on CO2 Emissions (Dataset Analysis):**\n"
            for rate_range, emissions in recycling_impact.items():
                response += f"- Recycling Rate {rate_range}: {emissions:,.2f} Metric Tons\n"
            
            if web_info:
                response += f"\n**Global Recycling Strategies (Web Data):**\n{web_info}\n"
            
            return response
        except Exception as e:
            print(f"Error in recycling analysis: {str(e)}")
            return None

    def get_ewaste_recycling_strategies(self):
        try:
            response = "**E-waste Recycling Strategies Analysis:**\n\n"
            
            # 1. Product-wise Analysis
            df = pd.DataFrame(self.ewaste_data)
            df['weight_kg'] = pd.to_numeric(df['weight_kg'], errors='coerce')
            df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
            df['recycled_price_usd'] = pd.to_numeric(df['recycled_price_usd'], errors='coerce')
            
            # Calculate total waste and recycling value by product type
            df['total_waste'] = df['weight_kg'] * df['quantity']
            df['recycling_value'] = df['recycled_price_usd'] * df['quantity']
            
            product_analysis = df.groupby('product_type').agg({
                'total_waste': 'sum',
                'recycling_value': 'sum',
                'quantity': 'sum'
            }).sort_values('total_waste', ascending=False)
            
            response += "**1. Product-wise E-waste Analysis:**\n"
            for product, data in product_analysis.head(5).iterrows():
                response += f"- {product}:\n"
                response += f"  * Total Waste: {data['total_waste']:,.2f} kg\n"
                response += f"  * Units: {data['quantity']:,.0f}\n"
                response += f"  * Recycling Value: ${data['recycling_value']:,.2f}\n"
            
            # 2. Get web data for best practices
            web_info = self.search_web("electronic waste recycling strategies best practices UNEP guidelines")
            if web_info:
                response += f"\n**2. Global Best Practices for E-waste Recycling:**\n{web_info}\n"
            
            # 3. Specific Recycling Recommendations
            response += "\n**3. Key E-waste Recycling Strategies:**\n"
            response += "1. **Collection and Segregation:**\n"
            response += "   - Establish dedicated e-waste collection points\n"
            response += "   - Separate devices by type (computers, phones, appliances)\n"
            response += "   - Remove batteries and hazardous components\n\n"
            
            response += "2. **Material Recovery:**\n"
            response += "   - Extract valuable metals (gold, silver, copper)\n"
            response += "   - Recycle plastic components\n"
            response += "   - Proper handling of circuit boards\n\n"
            
            response += "3. **Safe Disposal:**\n"
            response += "   - Proper treatment of hazardous materials\n"
            response += "   - Environmentally sound disposal methods\n"
            response += "   - Compliance with environmental regulations\n\n"
            
            response += "4. **Extended Producer Responsibility:**\n"
            response += "   - Manufacturer take-back programs\n"
            response += "   - Product design for easy recycling\n"
            response += "   - Incentives for recycling old devices\n\n"
            
            # 4. Economic Benefits
            total_recycling_value = df['recycling_value'].sum()
            response += "**4. Economic Benefits of E-waste Recycling:**\n"
            response += f"- Total Potential Recycling Value: ${total_recycling_value:,.2f}\n"
            response += "- Job creation in recycling sector\n"
            response += "- Recovery of valuable materials\n"
            response += "- Reduced environmental cleanup costs\n"
            
            return response
            
        except Exception as e:
            print(f"Error in e-waste recycling analysis: {str(e)}")
            return "I apologize, but I encountered an error while analyzing e-waste recycling strategies. Please try again."

    def get_response(self, user_message):
        try:
            if self.model is None or self.index is None or self.co2_data is None or self.ewaste_data is None:
                return "I apologize, but I'm having trouble accessing my knowledge base at the moment. Please try again later."

            user_message = user_message.lower()

            # Prioritize e-waste recycling queries
            if any(term in user_message for term in ["e-waste", "ewaste", "electronic waste"]) and \
               any(term in user_message for term in ["recycling", "recycle", "strategy", "strategies", "reduce", "reduction"]):
                ewaste_recycling_response = self.get_ewaste_recycling_strategies()
                if ewaste_recycling_response:
                    return ewaste_recycling_response

            # Handle CO2 recycling queries specifically
            if "co2" in user_message and ("recycling" in user_message or "strateg" in user_message):
                recycling_response = self.get_combined_recycling_strategies()
                if recycling_response:
                    return recycling_response

            # Handle sector analysis queries
            if "sector" in user_message or "industry" in user_message:
                sector_response = self.get_combined_sector_analysis()
                if sector_response:
                    return sector_response

            # Handle brand analysis queries
            if "brand" in user_message:
                brand_response = self.get_combined_brand_analysis()
                if brand_response:
                    return brand_response

            # Handle comprehensive analysis request
            if any(term in user_message for term in ["analysis", "overview", "summary", "comprehensive", "all", "both"]):
                response = "**Complete Environmental Impact Analysis:**\n\n"
                
                # Get all analyses
                sector_analysis = self.get_combined_sector_analysis()
                brand_analysis = self.get_combined_brand_analysis()
                recycling_analysis = self.get_combined_recycling_strategies()
                
                if sector_analysis:
                    response += f"{sector_analysis}\n\n"
                if brand_analysis:
                    response += f"{brand_analysis}\n\n"
                if recycling_analysis:
                    response += f"{recycling_analysis}\n"
                
                return response

            # Handle e-waste forecast queries
            if any(term in user_message for term in ["2025", "forecast", "future"]) and any(term in user_message for term in ["e-waste", "ewaste", "electronic waste"]):
                forecast_response = self.get_ewaste_forecast()
                if forecast_response:
                    return forecast_response

            # Handle CO2 forecast queries
            if any(term in user_message for term in ["2025", "forecast", "future"]) and "co2" in user_message:
                forecast_response = self.get_co2_forecast()
                if forecast_response:
                    return forecast_response

            # Handle specific queries first
            # 1. Recycling strategies query
            if "recycling" in user_message and ("strateg" in user_message or "reduce" in user_message):
                dataset_response = self.get_recycling_strategies()
                if dataset_response:
                    web_info = self.search_web("effective recycling strategies to reduce CO2 emissions")
                    if web_info:
                        return f"{dataset_response}\n\n**Additional Information from Web:**\n{web_info}"
                    return dataset_response

            # 2. Sector-wise CO2 emissions
            if "sector" in user_message and "co2" in user_message:
                df = pd.DataFrame(self.co2_data)
                df['CO2 Emissions (Metric Tons)'] = pd.to_numeric(df['CO2 Emissions (Metric Tons)'].astype(str).str.replace(',', ''), errors='coerce')
                sector_emissions = df.groupby('Sector')['CO2 Emissions (Metric Tons)'].sum().sort_values(ascending=False)
                
                response = "**CO2 Emissions by Sector:**\n\n"
                for sector, emissions in sector_emissions.items():
                    response += f"**{sector}:** {emissions:,.2f} Metric Tons\n"
                return response

            # 3. Brand-wise e-waste
            if "brand" in user_message and "e-waste" in user_message:
                df = pd.DataFrame(self.ewaste_data)
                df['weight_kg'] = pd.to_numeric(df['weight_kg'], errors='coerce')
                df['quantity'] = pd.to_numeric(df['quantity'], errors='coerce')
                df['total_waste'] = df['weight_kg'] * df['quantity']
                brand_waste = df.groupby('brand')['total_waste'].sum().sort_values(ascending=False)
                
                response = "**E-waste Generation by Brand:**\n\n"
                for brand, waste in brand_waste.head(5).items():
                    response += f"**{brand}:** {waste:,.2f} kg\n"
                return response

            # 4. Total e-waste query
            if "total e-waste" in user_message or "total ewaste" in user_message:
                total = self.calculate_total_ewaste()
                if total is not None:
                    return f"Based on our e-waste dataset, the total e-waste generated is approximately {total:,.2f} kg."

            # 5. CO2 emissions by year
            if "co2" in user_message and ("year" in user_message or "emission" in user_message):
                yearly_data = self.get_co2_emissions_by_year()
                if yearly_data:
                    response = "**CO2 Emissions by Year:**\n\n"
                    for year, emissions in sorted(yearly_data.items()):
                        response += f"**{year}:** {emissions:,.2f} Metric Tons\n"
                    response += "\n*Note: Data is based on our CO2 emissions dataset.*"
                    return response

            # 6. Specific brand query
            for brand in ["panasonic", "samsung", "apple", "dell", "hp", "lenovo"]:
                if brand in user_message:
                    total = self.calculate_brand_ewaste(brand)
                    if total is not None:
                        return f"Based on our e-waste dataset, {brand.title()} has generated approximately {total:,.2f} kg of e-waste."

            # Use FAISS for semantic search with web augmentation
            query_embedding = self.model.encode([user_message])
            distances, indices = self.index.search(np.array(query_embedding).astype('float32'), k=3)
            
            if distances[0][0] < 1.0:
                dataset_response = "Based on our environmental data:\n\n"
                for idx in indices[0]:
                    if idx < len(self.data_texts):
                        dataset_response += f"{self.data_texts[idx].strip()}\n\n"
                
                web_info = self.search_web(user_message)
                if web_info:
                    return f"{dataset_response}\n\n**Additional Information from Web:**\n{web_info}"
                return dataset_response
            
            web_info = self.search_web(user_message)
            if web_info:
                return f"Based on available information:\n\n{web_info}"
            
            return "I apologize, but I couldn't find specific information about that in our database or from web sources. Please try rephrasing your question or ask about CO2 emissions, e-waste, or specific brands."

        except Exception as e:
            print(f"Error getting response: {str(e)}")
            return "I apologize, but I encountered an error while processing your request. Please try again."

# Initialize a single instance of the chatbot
chatbot_instance = EcoMetricsChatbot()
