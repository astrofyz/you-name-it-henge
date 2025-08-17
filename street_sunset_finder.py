#!/usr/bin/env python3
"""
Street Sunset Finder

This script finds streets where the sun sets directly at the end of the street
on a given date. It uses geospatial analysis and astronomical calculations.
"""

import os
import sys
from datetime import datetime, date
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
import plotly.graph_objects as go
import plotly.express as px
from astropy.coordinates import EarthLocation, AltAz, get_sun
from astropy.time import Time
from astropy import units as u
import warnings

warnings.filterwarnings('ignore')

class StreetSunsetFinder:
    """Main class for finding streets where the sun sets directly at the end."""
    
    def __init__(self, latitude, longitude, target_date=None, azimuth_tolerance=5):
        """
        Initialize the StreetSunsetFinder.
        
        Parameters:
        -----------
        latitude : float
            Latitude of the location in decimal degrees
        longitude : float
            Longitude of the location in decimal degrees
        target_date : datetime.date, optional
            Date for sunset calculation (defaults to today)
        azimuth_tolerance : float, optional
            Tolerance in degrees for azimuth matching (default: 5)
        """
        self.latitude = latitude
        self.longitude = longitude
        self.target_date = target_date or date.today()
        self.azimuth_tolerance = azimuth_tolerance
        
        # Create Earth location for astronomical calculations
        self.location = EarthLocation(lat=latitude*u.deg, lon=longitude*u.deg)
        
        # Common pedestrian street types (modify based on your data)
        self.pedestrian_street_types = [
            'pedestrian', 'footway', 'path', 'walkway', 'esplanade',
            'promenade', 'boardwalk', 'trail', 'sidewalk'
        ]
        
    def calculate_sunset_azimuth(self):
        """
        Calculate the azimuth of the sun at sunset for the target date.
        
        Returns:
        --------
        float : Sunset azimuth in degrees
        """
        # Create time object for the target date at sunset (approximate)
        sunset_time = Time(f"{self.target_date} 18:00:00")  # Approximate sunset time
        
        # Get sun position
        sun = get_sun(sunset_time)
        
        # Transform to horizontal coordinates
        altaz = sun.transform_to(AltAz(obstime=sunset_time, location=self.location))
        
        # Return azimuth in degrees
        return altaz.az.deg
    
    def load_street_data(self, shapefile_path):
        """
        Load street data from shapefile and transform to geographic coordinates.
        
        Parameters:
        -----------
        shapefile_path : str
            Path to the shapefile containing street data
            
        Returns:
        --------
        geopandas.GeoDataFrame : Street data in WGS84 (lat/lon) coordinates
        """
        try:
            gdf = gpd.read_file(shapefile_path)
            print(f"Loaded {len(gdf)} street segments from {shapefile_path}")
            print(f"Original CRS: {gdf.crs}")
            
            # Show sample coordinates before transformation
            if len(gdf) > 0:
                sample_geom = gdf.iloc[0].geometry
                if sample_geom.geom_type == 'LineString':
                    coords = list(sample_geom.coords)
                    print(f"Sample coordinates (before transform): {coords[0]} to {coords[-1]}")
            
            # Transform to WGS84 (lat/lon) if not already
            if gdf.crs != 'EPSG:4326':
                print(f"Transforming from {gdf.crs} to EPSG:4326 (WGS84)...")
                gdf = gdf.to_crs('EPSG:4326')
                print("Transformation completed")
                
                # Show sample coordinates after transformation
                if len(gdf) > 0:
                    sample_geom = gdf.iloc[0].geometry
                    if sample_geom.geom_type == 'LineString':
                        coords = list(sample_geom.coords)
                        print(f"Sample coordinates (after transform): {coords[0]} to {coords[-1]}")
            else:
                print("Data already in WGS84 (lat/lon) coordinates")
            
            return gdf
        except Exception as e:
            print(f"Error loading shapefile: {e}")
            return None
    
    def filter_pedestrian_streets(self, gdf, street_type_column='highway'):
        """
        Filter for pedestrian streets.
        
        Parameters:
        -----------
        gdf : geopandas.GeoDataFrame
            Street data
        street_type_column : str
            Column name containing street type information
            
        Returns:
        --------
        geopandas.GeoDataFrame : Filtered pedestrian streets
        """
        if street_type_column not in gdf.columns:
            print(f"Warning: Column '{street_type_column}' not found. Using all streets.")
            return gdf
        
        # Filter for pedestrian streets
        pedestrian_mask = gdf[street_type_column].str.lower().isin(
            [t.lower() for t in self.pedestrian_street_types]
        )
        
        filtered_gdf = gdf[pedestrian_mask].copy()
        print(f"Found {len(filtered_gdf)} pedestrian streets")
        
        return filtered_gdf
    
    def calculate_street_azimuth(self, geometry):
        """
        Calculate the azimuth of a street (direction from start to end).
        Uses great circle bearing for geographic coordinates.
        
        Parameters:
        -----------
        geometry : shapely.geometry.LineString
            Street geometry in WGS84 (lat/lon) coordinates
            
        Returns:
        --------
        float : Street azimuth in degrees (0-360, where 0 is North, 90 is East)
        """
        if geometry.geom_type != 'LineString':
            return None
        
        coords = list(geometry.coords)
        if len(coords) < 2:
            return None
        
        # Get start and end points (lon, lat)
        start_lon, start_lat = coords[0]
        end_lon, end_lat = coords[-1]
        
        # Convert to radians
        start_lat_rad = np.radians(start_lat)
        start_lon_rad = np.radians(start_lon)
        end_lat_rad = np.radians(end_lat)
        end_lon_rad = np.radians(end_lon)
        
        # Calculate great circle bearing
        dlon = end_lon_rad - start_lon_rad
        
        y = np.sin(dlon) * np.cos(end_lat_rad)
        x = np.cos(start_lat_rad) * np.sin(end_lat_rad) - np.sin(start_lat_rad) * np.cos(end_lat_rad) * np.cos(dlon)
        
        # Calculate bearing in radians
        bearing_rad = np.arctan2(y, x)
        
        # Convert to degrees and normalize to 0-360
        azimuth = np.degrees(bearing_rad)
        azimuth = (azimuth + 360) % 360
        
        return azimuth
    
    def find_sunset_streets(self, gdf):
        """
        Find streets where the sun sets directly at the end.
        
        Parameters:
        -----------
        gdf : geopandas.GeoDataFrame
            Street data
            
        Returns:
        --------
        geopandas.GeoDataFrame : Streets matching sunset azimuth
        """
        sunset_azimuth = self.calculate_sunset_azimuth()
        print(f"Sunset azimuth for {self.target_date}: {sunset_azimuth:.1f}°")
        
        # Calculate azimuth for each street
        gdf = gdf.copy()
        gdf['street_azimuth'] = gdf.geometry.apply(self.calculate_street_azimuth)
        
        # Remove streets with invalid azimuth
        gdf = gdf.dropna(subset=['street_azimuth'])
        
        # Find streets matching sunset azimuth
        # We need to check both directions (street could face either way)
        azimuth_diff = np.abs(gdf['street_azimuth'] - sunset_azimuth)
        azimuth_diff_opposite = np.abs(gdf['street_azimuth'] - (sunset_azimuth + 180) % 360)
        
        # Take the minimum difference (either direction)
        min_diff = np.minimum(azimuth_diff, azimuth_diff_opposite)
        
        # Filter by tolerance
        matching_mask = min_diff <= self.azimuth_tolerance
        matching_streets = gdf[matching_mask].copy()
        
        # Add azimuth difference for reference
        matching_streets['azimuth_diff'] = min_diff[matching_mask]
        
        print(f"Found {len(matching_streets)} streets matching sunset azimuth")
        
        return matching_streets
    
    def create_interactive_map(self, all_streets, sunset_streets, output_path='sunset_streets_map.html'):
        """
        Create an interactive map showing all streets and highlighting sunset streets.
        
        Parameters:
        -----------
        all_streets : geopandas.GeoDataFrame
            All street data
        sunset_streets : geopandas.GeoDataFrame
            Streets matching sunset azimuth
        output_path : str
            Path for the output HTML file
        """
        # Create the map
        fig = go.Figure()
        
        # Add all streets (gray)
        if len(all_streets) > 0:
            for idx, row in all_streets.iterrows():
                coords = list(row.geometry.coords)
                lons = [coord[0] for coord in coords]
                lats = [coord[1] for coord in coords]
                
                fig.add_trace(go.Scattermapbox(
                    lon=lons,
                    lat=lats,
                    mode='lines',
                    line=dict(width=1, color='lightgray'),
                    name='All Streets',
                    showlegend=False,
                    hoverinfo='skip'
                ))
        
        # Add sunset streets (red)
        if len(sunset_streets) > 0:
            for idx, row in sunset_streets.iterrows():
                coords = list(row.geometry.coords)
                lons = [coord[0] for coord in coords]
                lats = [coord[1] for coord in coords]
                
                # Create hover text
                hover_text = f"Street Azimuth: {row['street_azimuth']:.1f}°<br>"
                hover_text += f"Azimuth Diff: {row['azimuth_diff']:.1f}°"
                
                fig.add_trace(go.Scattermapbox(
                    lon=lons,
                    lat=lats,
                    mode='lines',
                    line=dict(width=3, color='red'),
                    name='Sunset Streets',
                    text=hover_text,
                    hoverinfo='text'
                ))
        
        # Update layout
        fig.update_layout(
            title=f"Streets Where Sun Sets Directly at the End<br>{self.target_date}",
            mapbox=dict(
                style='open-street-map',
                center=dict(lat=self.latitude, lon=self.longitude),
                zoom=13
            ),
            height=600,
            showlegend=True
        )
        
        # Save the map
        fig.write_html(output_path)
        print(f"Interactive map saved to: {output_path}")
        
        return fig
    
    def export_results(self, sunset_streets, output_path='sunset_streets.csv'):
        """
        Export matching streets to CSV.
        
        Parameters:
        -----------
        sunset_streets : geopandas.GeoDataFrame
            Streets matching sunset azimuth
        output_path : str
            Path for the output CSV file
        """
        if len(sunset_streets) > 0:
            # Convert to regular DataFrame for CSV export
            df_export = sunset_streets.drop(columns=['geometry']).copy()
            df_export.to_csv(output_path, index=False)
            print(f"Results exported to: {output_path}")
        else:
            print("No matching streets to export")


def main():
    """Main function to run the street sunset analysis."""
    
    # Configuration
    # Example coordinates for San Francisco
    LATITUDE = 37.7749
    LONGITUDE = -122.4194
    TARGET_DATE = date(2024, 6, 21)  # Summer solstice
    AZIMUTH_TOLERANCE = 5  # degrees
    
    # Create data directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    # Initialize the finder
    finder = StreetSunsetFinder(
        latitude=LATITUDE,
        longitude=LONGITUDE,
        target_date=TARGET_DATE,
        azimuth_tolerance=AZIMUTH_TOLERANCE
    )
    
    print("Street Sunset Finder")
    print("=" * 50)
    print(f"Location: {LATITUDE}, {LONGITUDE}")
    print(f"Target Date: {TARGET_DATE}")
    print(f"Azimuth Tolerance: {AZIMUTH_TOLERANCE}°")
    print()
    
    # Check for shapefile
    shapefile_path = 'data/streets.shp'
    if not os.path.exists(shapefile_path):
        print(f"Shapefile not found at: {shapefile_path}")
        print("Please place your street shapefile in the data/ directory.")
        print("The file should be named 'streets.shp' or update the path in the script.")
        return
    
    # Load street data
    print("Loading street data...")
    all_streets = finder.load_street_data(shapefile_path)
    if all_streets is None:
        return
    
    # Filter pedestrian streets
    print("Filtering pedestrian streets...")
    pedestrian_streets = finder.filter_pedestrian_streets(all_streets)
    
    # Find sunset streets
    print("Finding streets matching sunset azimuth...")
    sunset_streets = finder.find_sunset_streets(pedestrian_streets)
    
    # Create interactive map
    print("Creating interactive map...")
    fig = finder.create_interactive_map(pedestrian_streets, sunset_streets)
    
    # Export results
    finder.export_results(sunset_streets)
    
    print("\nAnalysis complete!")
    print(f"Found {len(sunset_streets)} streets where the sun sets directly at the end.")
    print("Open 'sunset_streets_map.html' in your browser to view the interactive map.")


if __name__ == "__main__":
    main()
