# Slicer OME-Zarr Extension

A 3D Slicer extension for loading and visualizing OME-Zarr datasets.

## Features
- Load OME-Zarr datasets from S3 URLs
- Multi-resolution pyramid support
- Channel and timepoint selection
- Proper voxel spacing for accurate 3D visualization

## Installation
1. Install 3D Slicer
2. Install this extension
3. Dependencies (zarr, ome-zarr) will be installed automatically

## Usage
1. Open SlicerZarrOME module
2. Enter OME-Zarr URL or file path
3. Click "Connect & Analyze"
4. Select resolution level, channel, and timepoint
5. Click "Load Data"

## Requirements
- 3D Slicer 5.9.0 (required)
- zarr == 3.1.3 (used)
- ome-zarr == 0.12.2 (used)
