# Slicer OME-Zarr Extension

A 3D Slicer extension for loading and visualizing OME-Zarr datasets.

## Features
- Load OME-Zarr datasets from S3 URLs
- Multi-resolution pyramid support
- Channel and timepoint selection

## Installation
1. Install 3D Slicer 5.9.0 Preview Release
2. Click on Code(top-right) & Download Zip
3. Extract or unzip SlicerZarrOME-main.zip
4. Go to Welcome to slicer -> Developer Tools -> Extension Wizard
5. Click on select extension
6. Select SlicerZarrOME from the downloaded folder. Click Yes.
7. Go to Edit -> Application Settings -> Modules.
8. Select SlicerZarrOME and drag it to Favourite Modules. 

## Usage
1. Open SlicerZarrOME module - Click on Zarr logo.
2. Enter OME-Zarr URL or file path
3. Click "Connect & Analyze"
4. Select resolution level, channel, and timepoint
5. Click "Load Data"

## Requirements
- 3D Slicer 5.9.0 (required)
- zarr == 3.1.3 (used)
- ome-zarr == 0.12.2 (used)
