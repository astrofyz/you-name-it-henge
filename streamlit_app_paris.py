#!/usr/bin/env python3
"""
Street Sunset Finder - Paris Version

This Streamlit app displays a map and calendar interface for exploring
streets in Paris where the sun sets directly at the end on specific dates.
"""

import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime, date
import os
import io
from shapely.geometry import Point, LineString
import warnings
from shapely import wkt
import requests

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Paris Street Sunset Finder",
    page_icon="🗼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 1rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.2rem;
        margin-top: 0.1rem;
    }
    .date-info {
        background-color: #f0f2f6;
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin: 0.3rem 0;
    }
    .compact-calendar {
        font-size: 0.8rem;
    }
    .compact-button {
        font-size: 0.7rem;
        padding: 0.2rem 0.4rem;
    }
    .stInfo {
        margin-top: 0.2rem !important;
        margin-bottom: 0.2rem !important;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_paris_data():
    """
    Load Paris street data from URL or create sample data.
    
    Returns:
    --------
    geopandas.GeoDataFrame : Paris street data with geometry
    """
    try:
        # Try to load from URL (replace with your actual URL)
        paris_data_url = "https://raw.githubusercontent.com/yourusername/hedge/main/data/paris_streets.csv"
        
        # For now, create sample Paris data
        sample_data = {
            'Route_Name': [
                'Champs-Élysées',
                'Rue de Rivoli', 
                'Avenue des Champs-Élysées',
                'Boulevard Saint-Germain',
                'Rue de la Paix',
                'Avenue Montaigne',
                'Rue du Faubourg Saint-Honoré',
                'Boulevard Haussmann',
                'Rue de Sèvres',
                'Avenue Victor Hugo'
            ],
            'geometry': [
                'LINESTRING(2.3080 48.8738, 2.3090 48.8748)',
                'LINESTRING(2.3320 48.8588, 2.3330 48.8598)',
                'LINESTRING(2.3080 48.8738, 2.3090 48.8748)',
                'LINESTRING(2.3320 48.8588, 2.3330 48.8598)',
                'LINESTRING(2.3320 48.8688, 2.3330 48.8698)',
                'LINESTRING(2.3080 48.8638, 2.3090 48.8648)',
                'LINESTRING(2.3220 48.8688, 2.3230 48.8698)',
                'LINESTRING(2.3320 48.8738, 2.3330 48.8748)',
                'LINESTRING(2.3180 48.8488, 2.3190 48.8498)',
                'LINESTRING(2.2880 48.8638, 2.2890 48.8648)'
            ],
            'henge_time': [
                "['2024-06-21T21:30:00', '2024-12-21T16:45:00']",
                "['2024-06-21T21:30:00', '2024-12-21T16:45:00']",
                "['2024-06-21T21:30:00']",
                "['2024-12-21T16:45:00']",
                "['2024-06-21T21:30:00', '2024-12-21T16:45:00']",
                "['2024-06-21T21:30:00']",
                "['2024-12-21T16:45:00']",
                "['2024-06-21T21:30:00', '2024-12-21T16:45:00']",
                "['2024-06-21T21:30:00']",
                "['2024-12-21T16:45:00']"
            ],
            'azimuth': [300.5, 310.2, 295.8, 320.1, 305.7, 290.3, 315.9, 300.1, 285.6, 325.4]
        }
        
        df = pd.DataFrame(sample_data)
        
        # Convert geometry strings to Shapely objects
        df['geometry'] = df['geometry'].apply(wkt.loads)
        gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
        
        # Convert henge_time list of strings to list of datetime objects
        def parse_henge_time_list(henge_time_str):
            try:
                if isinstance(henge_time_str, str):
                    clean_str = henge_time_str.strip('[]').replace("'", "").replace('"', '')
                    time_strings = [s.strip() for s in clean_str.split(',')]
                    return [pd.to_datetime(ts, format='%Y-%m-%dT%H:%M:%S') for ts in time_strings]
                else:
                    return []
            except Exception:
                return []
        
        gdf['henge_time'] = gdf['henge_time'].apply(parse_henge_time_list)
        
        return gdf
        
    except Exception as e:
        st.error(f"Error loading Paris data: {e}")
        return None

@st.cache_data
def load_street_data(csv_path):
    """
    Load street data from CSV file and convert to GeoDataFrame.
    
    Parameters:
    -----------
    csv_path : str
        Path to the CSV file containing street data
        
    Returns:
    --------
    geopandas.GeoDataFrame : Street data with geometry
    """
    try:
        # Read CSV file
        df = pd.read_csv(csv_path)
        
        # Check if geometry columns exist
        if 'geometry' in df.columns:
            # If geometry is already in WKT format
            df['geometry'] = df['geometry'].apply(wkt.loads)
            gdf = gpd.GeoDataFrame(df, geometry='geometry', crs='EPSG:4326')
        elif all(col in df.columns for col in ['start_lon', 'start_lat', 'end_lon', 'end_lat']):
            # Create LineString geometries from start/end coordinates
            geometries = []
            for _, row in df.iterrows():
                start_point = (row['start_lon'], row['start_lat'])
                end_point = (row['end_lon'], row['end_lat'])
                geometries.append(LineString([start_point, end_point]))
            
            gdf = gpd.GeoDataFrame(df, geometry=geometries, crs='EPSG:4326')
        else:
            st.error("CSV file must contain either 'geometry' column or 'start_lon', 'start_lat', 'end_lon', 'end_lat' columns")
            return None
        
        # Convert date columns to datetime
        if 'henge_time' in gdf.columns:
            # Convert henge_time list of strings to list of datetime objects
            def parse_henge_time_list(henge_time_str):
                try:
                    # Remove brackets and split by comma
                    if isinstance(henge_time_str, str):
                        # Remove quotes and brackets, split by comma
                        clean_str = henge_time_str.strip('[]').replace("'", "").replace('"', '')
                        time_strings = [s.strip() for s in clean_str.split(',')]
                        # Convert each string to datetime
                        return [pd.to_datetime(ts, format='%Y-%m-%dT%H:%M:%S') for ts in time_strings]
                    else:
                        return []
                except Exception:
                    return []
            
            gdf['henge_time'] = gdf['henge_time'].apply(parse_henge_time_list)
        elif 'henge_date' in gdf.columns:
            gdf['henge_date'] = pd.to_datetime(gdf['henge_date'], errors='coerce')
        
        return gdf
        
    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
        return None

def create_map(gdf, selected_date=None, center_lat=48.8566, center_lon=2.3522, window_height=None):
    """
    Create an interactive map using Plotly with efficient trace handling.
    
    Parameters:
    -----------
    gdf : geopandas.GeoDataFrame
        Street data
    selected_date : datetime.date, optional
        Selected date to highlight streets
    center_lat : float
        Center latitude for the map
    center_lon : float
        Center longitude for the map
    window_height : int, optional
        Window height to calculate map height
        
    Returns:
    --------
    plotly.graph_objects.Figure : Interactive map
    """
    if gdf is None or len(gdf) == 0:
        # Create empty map
        fig = go.Figure()
        fig.update_layout(
            title="No street data available",
            xaxis_title="Longitude",
            yaxis_title="Latitude"
        )
        return fig
    
    # Get map bounds from data
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    
    # Prepare data for efficient plotting
    regular_lats = []
    regular_lons = []
    regular_names = []
    highlighted_lats = []
    highlighted_lons = []
    highlighted_names = []
    
    # Process each street
    for idx, row in gdf.iterrows():
        # Check if this street matches the selected date
        is_highlighted = False
        if selected_date:
            # Check henge_time column first, then henge_date
            if 'henge_time' in row and isinstance(row['henge_time'], list) and len(row['henge_time']) > 0:
                # Check if any date in the henge_time list matches the selected date
                henge_dates = [dt.date() for dt in row['henge_time'] if pd.notna(dt)]
                is_highlighted = selected_date in henge_dates
            elif 'henge_date' in row and pd.notna(row['henge_date']):
                street_date = row['henge_date'].date()
                is_highlighted = street_date == selected_date
        
        # Get street name for hover
        street_name = row.get('Route_Name', row.get('name', f'Street {idx}'))
        if pd.isna(street_name):
            street_name = f'Street {idx}'
        
        # Create styled hover text
        if is_highlighted:
            hover_text = f'<span style="color: rgba(255, 0, 0, 0.8); font-weight: bold;">{street_name}</span>'
        else:
            hover_text = f"<b>{street_name}</b>"
        
        # Add coordinates to appropriate lists
        if row.geometry.geom_type == 'LineString':
            coords = list(row.geometry.coords)
            lons = [coord[0] for coord in coords]
            lats = [coord[1] for coord in coords]
            
            if is_highlighted:
                highlighted_lats.extend(lats)
                highlighted_lons.extend(lons)
                highlighted_names.extend([hover_text] * len(lats))
                # Add None to separate lines
                highlighted_lats.append(None)
                highlighted_lons.append(None)
                highlighted_names.append(None)
            else:
                regular_lats.extend(lats)
                regular_lons.extend(lons)
                regular_names.extend([hover_text] * len(lats))
                # Add None to separate lines
                regular_lats.append(None)
                regular_lons.append(None)
                regular_names.append(None)
    
    # Create figure
    fig = go.Figure()
    
    # Add regular streets trace (if any)
    if regular_lats:
        fig.add_trace(go.Scatter(
            x=regular_lons,
            y=regular_lats,
            mode='lines',
            line=dict(color='lightgrey', width=1),
            hoverinfo='text',
            hovertext=regular_names,
            showlegend=False,
            name='Regular Streets',
            opacity=0.6
        ))
    
    # Add highlighted streets trace (if any)
    if highlighted_lats:
        fig.add_trace(go.Scatter(
            x=highlighted_lons,
            y=highlighted_lats,
            mode='lines',
            line=dict(color='red', width=3),
            hoverinfo='text',
            hovertext=highlighted_names,
            showlegend=False,
            name='Highlighted Streets',
            opacity=0.8
        ))
    
    # Calculate bounds with padding
    x_range = bounds[2] - bounds[0]
    y_range = bounds[3] - bounds[1]
    
    # Add 10% padding to the bounds
    padding_x = x_range * 0.0
    padding_y = y_range * 0.0
    
    # Set ranges with natural proportions
    x_min = bounds[0] - padding_x
    x_max = bounds[2] + padding_x
    y_min = bounds[1] - padding_y
    y_max = bounds[3] + padding_y

    y_center = (y_min + y_max) / 2
    
    # Update layout
    fig.update_layout(
        title=f"Paris Street Map - {selected_date.strftime('%B %d, %Y') if selected_date else 'All Streets'}",
        xaxis=dict(
            range=[x_min, x_max],
            showgrid=False,
            showticklabels=False,
            showline=False,
            zeroline=False
        ),
        yaxis=dict(
            range=[y_min, y_max],
            showgrid=False,
            showticklabels=False,
            showline=False,
            zeroline=False
        ),
        height=window_height if window_height else 500,
        hovermode='closest',
        margin=dict(l=0, r=0, t=30, b=0)
    )

    fig.update_layout({'xaxis': {'scaleanchor': 'y', 'scaleratio': np.cos(y_center*np.pi/180)}})
    
    return fig

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🗼 Paris Street Sunset Finder</h1>', unsafe_allow_html=True)
    
    # Sidebar for file uploads and configuration
    with st.sidebar:
        st.header("📁 Data Configuration")
        
        # File upload section
        st.subheader("Upload Data Files")
        
        # CSV file upload (optional)
        csv_file = st.file_uploader(
            "Upload Custom Street Data (CSV) - Optional",
            type=['csv'],
            help="Upload a CSV file to replace the default Paris data"
        )

    
    # Main content area
    if csv_file is not None:
        # Load custom street data
        with st.spinner("Loading custom street data..."):
            gdf = load_street_data(csv_file)
    else:
        # Load default Paris data
        with st.spinner("Loading Paris street data..."):
            gdf = load_paris_data()
    
    if gdf is not None:
        # Display data info
        st.success(f"✅ Loaded {len(gdf)} street segments")
        
        # Main layout with map and calendar
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.subheader("🗺️ Interactive Map")
            
            # Date selection
            selected_date = st.date_input(
                "Select Date:",
                value=date.today(),
                help="Choose a date to highlight matching streets"
            )
            
            # Create and display map
            # Calculate map height (85% of typical window height minus header space)
            map_height = int(0.85 * 800) - 100  # Assuming 800px window, 100px for header/calendar
            map_obj = create_map(gdf, selected_date, 48.8566, 2.3522, map_height)
            st.plotly_chart(map_obj, use_container_width=True)
        
        with col2:
            st.subheader("📅 Date Information")
            
            if selected_date:
                # Find highlighted streets for this date
                highlighted_streets = []
                sunset_time = None
                
                for idx, row in gdf.iterrows():
                    is_highlighted = False
                    if 'henge_time' in row and isinstance(row['henge_time'], list) and len(row['henge_time']) > 0:
                        henge_dates = [dt.date() for dt in row['henge_time'] if pd.notna(dt)]
                        is_highlighted = selected_date in henge_dates
                        if is_highlighted and sunset_time is None:
                            # Find the matching time for this date
                            for dt in row['henge_time']:
                                if pd.notna(dt) and dt.date() == selected_date:
                                    sunset_time = dt
                                    break
                    elif 'henge_date' in row and pd.notna(row['henge_date']):
                        street_date = row['henge_date'].date()
                        is_highlighted = street_date == selected_date
                    
                    if is_highlighted:
                        street_name = row.get('Route_Name', row.get('name', f'Street {idx}'))
                        highlighted_streets.append(street_name)
                
                # Display selected date info
                st.markdown(f"""
                <div class="date-info" style="padding: 0.5rem; margin: 0.5rem 0;">
                    <h5 style="margin: 0;">{selected_date.strftime('%B %d, %Y')}</h5>
                    <small>Day {selected_date.timetuple().tm_yday}</small>
                </div>
                """, unsafe_allow_html=True)
                
                # Add sunset time if found
                if sunset_time:
                    st.markdown("**🌅 Sunset Time**")
                    st.markdown(f"**{sunset_time.strftime('%H:%M:%S')}**")
                
                # Check if data has required columns
                if 'henge_time' not in gdf.columns and 'henge_date' not in gdf.columns:
                    st.warning("No 'henge_time' or 'henge_date' column found in data")

    
    else:
        # Welcome screen when no data is loaded
        st.info("👆 Please upload a CSV file with street data to get started")
        
        # Show example data format
        with st.expander("📋 Expected Data Format"):
            st.markdown("""
            Your CSV file should contain one of these formats:
            
            **Option 1: Geometry column**
            ```csv
            Route_Name,geometry,henge_time,azimuth
            "Champs-Élysées","LINESTRING(2.3080 48.8738, 2.3090 48.8748)","['2024-06-21T21:30:00', '2024-12-21T16:45:00']",300.5
            ```
            
            **Option 2: Start/End coordinates**
            ```csv
            Route_Name,start_lon,start_lat,end_lon,end_lat,henge_time,azimuth
            "Champs-Élysées",2.3080,48.8738,2.3090,48.8748,"['2024-06-21T21:30:00', '2024-12-21T16:45:00']",300.5
            ```
            
            **Note:** 
            - The `henge_time` column should contain a list of datetime strings in format: `['YYYY-MM-DDTHH:MM:SS', 'YYYY-MM-DDTHH:MM:SS', ...]`
            - The `Route_Name` column is used for hover information on the map
            """)

if __name__ == "__main__":
    main()
