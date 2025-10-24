"""
OME-Zarr Extension for 3D Slicer

"""

import logging

import os
from typing import Annotated

import vtk
import numpy

import slicer
from slicer.i18n import tr as _
from slicer.i18n import translate
from slicer.ScriptedLoadableModule import *
from slicer.util import VTKObservationMixin
from slicer.parameterNodeWrapper import (
    parameterNodeWrapper,
    WithinRange,
)

from slicer import vtkMRMLScalarVolumeNode



def ensureOMEZarrAvailable():
    """Install zarr v3.1.3 and ome-zarr v0.12.2 every time"""
    global OME_ZARR_AVAILABLE, ome_zarr, parse_url, Reader
    
    try:
        print("Installing zarr v3.1.3 and ome-zarr v0.12.2...")
        
        # Install exact versions
        slicer.util.pip_install("zarr==3.1.3")
        slicer.util.pip_install("ome-zarr==0.12.2")
        
        # Import modules
        import ome_zarr
        from ome_zarr.io import parse_url
        from ome_zarr.reader import Reader
        
        OME_ZARR_AVAILABLE = True
        print("✅ Dependencies installed successfully")
        return True
        
    except Exception as e:
        logging.error(f"Failed to install dependencies: {e}")
        print(f"❌ Installation failed: {e}")
        return False



#
# SlicerZarrOME
#


class SlicerZarrOME(ScriptedLoadableModule):
    """Uses ScriptedLoadableModule base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("SlicerZarrOME")  # TODO: make this more human readable by adding spaces
        # TODO: set categories (folders where the module shows up in the module selector)
        self.parent.categories = [translate("qSlicerAbstractCoreModule", "Examples")]
        self.parent.dependencies = []  # TODO: add here list of module names that this module requires
        self.parent.contributors = ["Niraj Kandpal (NFDI4BIOIMAGE)"]  # TODO: replace with "Firstname Lastname (Organization)"
        # TODO: update with short description of the module and a link to online module documentation
        # _() function marks text as translatable to other languages
        self.parent.helpText = _("""
This is an example of scripted loadable module bundled in an extension.
See more information in <a href="https://github.com/organization/projectname#SlicerZarrOME">module documentation</a>.
""")
        # TODO: replace with organization, grant and thanks
        self.parent.acknowledgementText = _("""
                                            This file is modified and developed as a part of the NFDI4BIOIMAGE project.
""")

        # Additional initialization step after application startup is complete
        #slicer.app.connect("startupCompleted()", registerSampleData)


#
# Register sample data sets in Sample Data module
#

'''
def registerSampleData():
    """Add data sets to Sample Data module."""
    # It is always recommended to provide sample data for users to make it easy to try the module,
    # but if no sample data is available then this method (and associated startupCompeted signal connection) can be removed.

    import SampleData

    iconsPath = os.path.join(os.path.dirname(__file__), "Resources/Icons")

    # To ensure that the source code repository remains small (can be downloaded and installed quickly)
    # it is recommended to store data sets that are larger than a few MB in a Github release.

    # SlicerZarrOME1
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="SlicerZarrOME",
        sampleName="SlicerZarrOME1",
        # Thumbnail should have size of approximately 260x280 pixels and stored in Resources/Icons folder.
        # It can be created by Screen Capture module, "Capture all views" option enabled, "Number of images" set to "Single".
        thumbnailFileName=os.path.join(iconsPath, "SlicerZarrOME1.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        fileNames="SlicerZarrOME1.nrrd",
        # Checksum to ensure file integrity. Can be computed by this command:
        #  import hashlib; print(hashlib.sha256(open(filename, "rb").read()).hexdigest())
        checksums="SHA256:998cb522173839c78657f4bc0ea907cea09fd04e44601f17c82ea27927937b95",
        # This node name will be used when the data set is loaded
        nodeNames="SlicerZarrOME1",
    )

    # SlicerZarrOME2
    SampleData.SampleDataLogic.registerCustomSampleDataSource(
        # Category and sample name displayed in Sample Data module
        category="SlicerZarrOME",
        sampleName="SlicerZarrOME2",
        thumbnailFileName=os.path.join(iconsPath, "SlicerZarrOME2.png"),
        # Download URL and target file name
        uris="https://github.com/Slicer/SlicerTestingData/releases/download/SHA256/1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        fileNames="SlicerZarrOME2.nrrd",
        checksums="SHA256:1a64f3f422eb3d1c9b093d1a18da354b13bcf307907c66317e2463ee530b7a97",
        # This node name will be used when the data set is loaded
        nodeNames="SlicerZarrOME2",
    )
    
    
    '''


#
# SlicerZarrOMEParameterNode
#


@parameterNodeWrapper
class SlicerZarrOMEParameterNode:
    """
    The parameters needed by OME-Zarr module.
    """
    # Data source
    zarrUrl: str = ""
    
    # Dataset metadata (populated after connection)
    dimensions: str = ""
    channelCount: int = 0
    timepointCount: int = 0
    resolutionLevels: int = 0
    
    # Loading configuration
    selectedResolution: Annotated[int, WithinRange(0, 20)] = 0
    selectedChannel: Annotated[int, WithinRange(0, 100)] = 0
    
 
#
# SlicerZarrOMEWidget
#


class SlicerZarrOMEWidget(ScriptedLoadableModuleWidget, VTKObservationMixin):
    """Uses ScriptedLoadableModuleWidget base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def __init__(self, parent=None) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.__init__(self, parent)
        VTKObservationMixin.__init__(self)  # needed for parameter node observation
        self.logic = None
        self._parameterNode = None
        self._parameterNodeGuiTag = None

    def setup(self) -> None:
        """Called when the user opens the module the first time and the widget is initialized."""
        ScriptedLoadableModuleWidget.setup(self)

        # Load widget from .ui file (created by Qt Designer).
        uiWidget = slicer.util.loadUI(self.resourcePath("UI/SlicerZarrOME.ui"))
        self.layout.addWidget(uiWidget)
        self.ui = slicer.util.childWidgetVariables(uiWidget)

        # Set scene in MRML widgets
        uiWidget.setMRMLScene(slicer.mrmlScene)

        # Create logic class
        self.logic = SlicerZarrOMELogic()

        # OME-Zarr specific button connections
        self.ui.connectButton.connect("clicked(bool)", self.onConnectClicked)
        self.ui.loadButton.connect("clicked(bool)", self.onLoadClicked)
        
        # URL input connection
        self.ui.urlLineEdit.connect("textChanged(QString)", self.onUrlChanged)
        
        # Scene connections (these are important to add)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.StartCloseEvent, self.onSceneStartClose)
        self.addObserver(slicer.mrmlScene, slicer.mrmlScene.EndCloseEvent, self.onSceneEndClose)

        # Initialize parameter node
        self.initializeParameterNode()

    def cleanup(self) -> None:
        """Called when the application closes and the module widget is destroyed."""
        self.removeObservers()

    def enter(self) -> None:
        """Called each time the user opens this module."""
        self.initializeParameterNode()

    def exit(self) -> None:
        """Called each time the user opens a different module."""
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self._parameterNodeGuiTag = None
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanConnect)

    def onSceneStartClose(self, caller, event) -> None:
        """Called just before the scene is closed."""
        self.setParameterNode(None)

    def onSceneEndClose(self, caller, event) -> None:
        """Called just after the scene is closed."""
        if self.parent.isEntered:
            self.initializeParameterNode()

    def initializeParameterNode(self) -> None:
        """Ensure parameter node exists and observed - OME-Zarr version."""
        self.setParameterNode(self.logic.getParameterNode())
        
    def setParameterNode(self, inputParameterNode: SlicerZarrOMEParameterNode | None) -> None:
        """Set and observe parameter node."""
        if self._parameterNode:
            self._parameterNode.disconnectGui(self._parameterNodeGuiTag)
            self.removeObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanConnect)
        
        self._parameterNode = inputParameterNode
        
        if self._parameterNode:
            self._parameterNodeGuiTag = self._parameterNode.connectGui(self.ui)
            self.addObserver(self._parameterNode, vtk.vtkCommand.ModifiedEvent, self._checkCanConnect)
            self._checkCanConnect()

    def _checkCanConnect(self, caller=None, event=None) -> None:
        """Check if connect button should be enabled."""
        if self._parameterNode and self._parameterNode.zarrUrl:
            self.ui.connectButton.enabled = True
            self.ui.connectButton.toolTip = _("Connect to OME-Zarr dataset")
        else:
            self.ui.connectButton.enabled = False
            #self.ui.connectButton.toolTip = _("Please enter a valid OME-Zarr URL")

    # =====================================================
    # OME-ZARR SPECIFIC EVENT HANDLERS
    # =====================================================

    def onUrlChanged(self, text):
        """Handle URL input change."""
        if self._parameterNode:
            self._parameterNode.zarrUrl = text.strip()
        
        # Reset UI state when URL changes
        self.resetUIState()

    def onConnectClicked(self):
        """Handle Connect & Analyze button click."""
        try:
            self.ui.statusValue.text = "Connecting..."
            self.ui.statusValue.styleSheet = "color: orange;"
            self.ui.connectButton.enabled = False

            # Force UI update to show immediately
            slicer.app.processEvents()
            
            # Get URL from UI
            url = self.ui.urlLineEdit.text.strip()
            if not url:
                raise ValueError("Please enter a valid OME-Zarr URL")
            
            # Check OME-Zarr availability
            if not ensureOMEZarrAvailable():
                raise RuntimeError("Failed to install OME-Zarr dependencies")
            
            # Connect and get metadata
            print("Attempting to connect to OME-Zarr dataset...")
            metadata = self.logic.connectToZarr(url)
            
            # Update UI with metadata
            self.ui.dimensionsValue.text = metadata['dimensions']
            self.ui.channelsValue.text = str(metadata['channelCount'])
            self.ui.timepointsValue.text = str(metadata['timepointCount'])
            self.ui.resolutionLevelsValue.text = str(metadata['resolutionLevels'])
            
            # Populate resolution dropdown with actual dimensions
            self.ui.resolutionComboBox.clear()
            node = self.logic.nodes[0]  # Get the node to access data shapes
            for i in range(metadata['resolutionLevels']):
                try:
                    # Get the shape for this resolution level
                    shape = node.data[i].shape
                    
                    # Extract spatial dimensions (last 3 dimensions are typically Z,Y,X)
                    if len(shape) >= 3:
                        z, y, x = shape[-3], shape[-2], shape[-1]
                        self.ui.resolutionComboBox.addItem(f"Level {i}: ({x}×{y}×{z})")
                    else:
                        # Fallback for 2D or unusual shapes
                        self.ui.resolutionComboBox.addItem(f"Level {i}: ({shape})")
                except Exception as e:
                    # Fallback if we can't get dimensions
                    self.ui.resolutionComboBox.addItem(f"Level {i} (Resolution)")
                    print(f"Warning: Could not get dimensions for level {i}: {e}")
            
            # Update channel spinbox (single channel only)
            max_channels = max(0, metadata['channelCount'] - 1)
            self.ui.channelSpinBox.maximum = max_channels
            self.ui.channelSpinBox.value = 0  # Default to first channel
            
            # Update timepoint spinbox 
            max_timepoints = max(0, metadata['timepointCount'] - 1)
            self.ui.TimepointSpinBox.maximum = max_timepoints
            self.ui.TimepointSpinBox.value = 0  # Default to first timepoint
            
            # Enable subsequent sections
            self.ui.datasetInfoCollapsibleButton.enabled = True
            self.ui.loadingConfigCollapsibleButton.enabled = True
            self.ui.loadButton.enabled = True
            
            # Update status
            self.ui.statusValue.text = "Connected successfully"
            self.ui.statusValue.styleSheet = "color: green;"
            
            print("Successfully connected to OME-Zarr dataset")
            
        except Exception as e:
            error_msg = str(e)
            self.ui.statusValue.text = f"Connection failed: {error_msg}"
            self.ui.statusValue.styleSheet = "color: red;"
            logging.error(f"Connection failed: {e}")
            
            # Show user-friendly error dialog
            slicer.util.errorDisplay(f"Failed to connect to OME-Zarr dataset:\n\n{error_msg}")
            
        finally:
            self.ui.connectButton.enabled = True


    def onLoadClicked(self):
        """Handle Load Data button click - loads full resolution data."""
        try:
            self.ui.progressLabel.text = "Loading OME-Zarr data..."
            self.ui.progressLabel.styleSheet = "color: blue;"
            self.ui.loadProgressBar.visible = True
            self.ui.loadProgressBar.value = 10
            
            # Disable buttons during loading
            self.ui.loadButton.enabled = False

            # Force UI update to show immediately
            slicer.app.processEvents()
            
            # Get parameters from UI
            url = self.ui.urlLineEdit.text.strip()
            resolution_level = self.ui.resolutionComboBox.currentIndex
            channel = self.ui.channelSpinBox.value
            
            # Get timepoint selection
            timepoint = self.ui.TimepointSpinBox.value
            
            self.ui.loadProgressBar.value = 30
            
            print(f"Loading OME-Zarr: level={resolution_level}, channel={channel}, timepoint={timepoint}")
            
            # Load the data using your original script logic
            volumeNode = self.logic.loadZarrData(url, resolution_level, channel, timepoint)
            
            self.ui.loadProgressBar.value = 80

            self.ui.loadProgressBar.value = 100
            self.ui.progressLabel.text = "Data loaded successfully!"
            self.ui.progressLabel.styleSheet = "color: green;"
            
            print(f"OME-Zarr data loaded successfully:")
            
            # Show success message
            slicer.util.infoDisplay(f"OME-Zarr data loaded successfully!\n\nVolume: \nResolution Level: {resolution_level}\nChannel: {channel}")
            
        except Exception as e:
            error_msg = str(e)
            self.ui.progressLabel.text = f"Load failed: {error_msg}"
            self.ui.progressLabel.styleSheet = "color: red;"
            logging.error(f"Load failed: {e}")
            
            # Show detailed error to user
            slicer.util.errorDisplay(f"Failed to load OME-Zarr data:\n\n{error_msg}")
            
        finally:
            # Re-enable buttons
            self.ui.loadButton.enabled = True
            self.ui.loadProgressBar.visible = False

    def resetUIState(self):
        """Reset UI to initial state when URL changes."""
        # Disable sections
        self.ui.datasetInfoCollapsibleButton.enabled = False
        self.ui.loadingConfigCollapsibleButton.enabled = False
        
        # Clear metadata
        self.ui.dimensionsValue.text = "-"
        self.ui.channelsValue.text = "-"
        self.ui.timepointsValue.text = "-"
        self.ui.resolutionLevelsValue.text = "-"
        
        # Clear dropdowns
        self.ui.resolutionComboBox.clear()

        # Reset spinboxes
        self.ui.channelSpinBox.maximum = 0
        self.ui.channelSpinBox.value = 0
        self.ui.TimepointSpinBox.maximum = 0
        self.ui.TimepointSpinBox.value = 0
        
        # Reset status
        self.ui.statusValue.text = "Not connected"
        self.ui.statusValue.styleSheet = "color: gray;"
        
        # Disable action buttons
        self.ui.loadButton.enabled = False
        
        # Reset progress
        self.ui.progressLabel.text = "Ready to connect"
        self.ui.progressLabel.styleSheet = "color: gray; font-style: italic;"
        self.ui.loadProgressBar.visible = False
        



    


#
# SlicerZarrOMELogic
#


class SlicerZarrOMELogic(ScriptedLoadableModuleLogic):
    """This class should implement all the actual
    computation done by your module.  The interface
    should be such that other python code can import
    this class and make use of the functionality without
    requiring an instance of the Widget.
    Uses ScriptedLoadableModuleLogic base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """
 
    def __init__(self) -> None:
        ScriptedLoadableModuleLogic.__init__(self)
        self.reader = None
        self.nodes = None

    def getParameterNode(self):
        return SlicerZarrOMEParameterNode(super().getParameterNode())

    def connectToZarr(self, url: str):
        """Connect to OME-Zarr dataset and read metadata"""
        if not ensureOMEZarrAvailable():
            raise RuntimeError("OME-Zarr dependencies not available")
        
        try:
            print(f"Connecting to OME-Zarr: {url}")
            
            # Setup connection
            self._setupConnection(url)
            node = self.nodes[0]
            
            # Extract and parse metadata
            data_shape = node.data[0].shape
            axes_info = self._parseAxesMetadata(node, data_shape)
            
            # Create final metadata
            metadata = self._buildMetadataResponse(data_shape, axes_info, node)
            
            print(f"Successfully connected to OME-Zarr dataset")
            print(f"Final metadata: {metadata}")
            return metadata
            
        except Exception as e:
            logging.error(f"Failed to connect to OME-Zarr: {e}")
            raise


    def _setupConnection(self, url):
        """Setup OME-Zarr reader and get nodes"""
        store = parse_url(url, mode="r").store
        print(f"DEBUG: Store type: {type(store)}")
        self.reader = Reader(parse_url(url))
        print(f"DEBUG: Reader type: {type(self.reader)}")
        self.nodes = [node for node in self.reader()]
        print(f"DEBUG: Found {len(self.nodes)} data nodes")

        if not self.nodes:
            raise ValueError("No data nodes found in OME-Zarr dataset")
        

    def _parseAxesMetadata(self, node, data_shape):
        """Parse axes information from OME-Zarr metadata"""
        # Initialize defaults
        axes_info = {
            'channels': 1,
            'timepoints': 1,
            'channel_axis': None,
            'time_axis': None,
            'spatial_axes': {},
            'axis_order': [],  # NEW: Store complete axis order
            'axis_names': []   # NEW: Store axis names for debugging
        }
        
        try:
            if hasattr(node, 'metadata') and node.metadata and 'axes' in node.metadata:
                axes = node.metadata['axes']
                print(f"Found axes metadata: {axes}")
                
                axes_info = self._processAxesList(axes, data_shape, axes_info)
                
                print(f"DEBUG - Complete axis order: {axes_info['axis_order']}")
                print(f"DEBUG - Axis names: {axes_info['axis_names']}")
                
                self._storeAxisInfo(axes_info)
                self._logSpatialSummary(axes_info['spatial_axes'])
                
            else:
                logging.warning("No axes metadata found, using shape-based heuristics")
                axes_info = self._fallbackAxisDetection(data_shape, axes_info)
                    
        except Exception as e:
            logging.error(f"Error reading axes metadata: {e}")
            raise

        return axes_info


    def _processAxesList(self, axes, data_shape, axes_info):
        """Process each axis in the axes list and capture complete order"""
        for i, axis in enumerate(axes):
            axis_name = axis.get("name", "")
            axis_type = axis.get("type", "")
            axis_unit = axis.get("unit", "")
            
            print(f"Axis {i}: name='{axis_name}', type='{axis_type}', unit='{axis_unit}'")
            
            # Store axis order information
            axes_info['axis_order'].append(axis_type)
            axes_info['axis_names'].append(axis_name)
            
            # Process different axis types
            if axis_type == "channel" or axis_name.lower() == "c":
                axes_info = self._processChannelAxis(i, data_shape, axes_info)
            elif axis_type == "time" or axis_name.lower() == "t":
                axes_info = self._processTimeAxis(i, data_shape, axes_info)
            elif axis_type == "space":
                axes_info = self._processSpatialAxis(i, axis_name, axis_unit, data_shape, axes_info)
        
        return axes_info
    

    def _processChannelAxis(self, index, data_shape, axes_info):
        """Process channel axis information"""
        axes_info['channel_axis'] = index
        axes_info['channels'] = data_shape[index]
        print(f"Found channel axis at index {index}: {axes_info['channels']} channels")
        return axes_info

    def _processTimeAxis(self, index, data_shape, axes_info):
        """Process time axis information"""
        axes_info['time_axis'] = index
        axes_info['timepoints'] = data_shape[index]
        print(f"Found time axis at index {index}: {axes_info['timepoints']} timepoints")
        return axes_info

    def _processSpatialAxis(self, index, axis_name, axis_unit, data_shape, axes_info):
        """Process spatial axis information"""
        axis_info = {
            'index': index,
            'name': axis_name,
            'unit': axis_unit,
            'size': data_shape[index]
        }
        
        axis_key = axis_name.lower()
        if axis_key in ['x', 'y', 'z']:
            axes_info['spatial_axes'][axis_key] = axis_info
            print(f"Found {axis_name} spatial axis: index={index}, size={data_shape[index]}, unit='{axis_unit}'")
        
        return axes_info

    def _storeAxisInfo(self, axes_info):
        """Store axis information as instance variables for use in loadZarrData"""
        self.channel_axis = axes_info['channel_axis']
        self.time_axis = axes_info['time_axis']
        self.spatial_axes = axes_info['spatial_axes']
        
        # Store complete axis order information
        self.axis_order = axes_info.get('axis_order', [])
        self.axis_names = axes_info.get('axis_names', [])
        
        # Store individual spatial axes for easy access
        self.x_axis_info = axes_info['spatial_axes'].get('x')
        self.y_axis_info = axes_info['spatial_axes'].get('y')
        self.z_axis_info = axes_info['spatial_axes'].get('z')
        
        print(f"DEBUG: Stored axis order: {self.axis_order}")
        print(f"DEBUG: Stored axis names: {self.axis_names}")

    def _fallbackAxisDetection(self, data_shape, axes_info):
        """Fallback axis detection using shape heuristics"""
        if len(data_shape) == 5:
            if data_shape[0] <= data_shape[1]:
                axes_info['channels'] = data_shape[0]
                axes_info['timepoints'] = data_shape[1]
                axes_info['channel_axis'] = 0
                axes_info['time_axis'] = 1
            else:
                axes_info['timepoints'] = data_shape[0]
                axes_info['channels'] = data_shape[1]
                axes_info['channel_axis'] = 1
                axes_info['time_axis'] = 0
        elif len(data_shape) == 4:
            if data_shape[0] < 10:
                axes_info['channels'] = data_shape[0]
                axes_info['channel_axis'] = 0
            else:
                axes_info['timepoints'] = data_shape[0]
                axes_info['time_axis'] = 0
    
        return axes_info
    

    
    def _logSpatialSummary(self, spatial_axes):
        """Log summary of spatial dimensions"""
        if spatial_axes:
            print("Spatial axes summary:")
            for axis_name, info in spatial_axes.items():
                print(f"  {axis_name.upper()}: size={info['size']}, unit='{info['unit']}'")

    def _buildMetadataResponse(self, data_shape, axes_info, node):
        """Build the final metadata response dictionary"""
        return {
            'dimensions': str(data_shape),
            'channelCount': axes_info.get('channels', 1),  # Use .get() with default
            'timepointCount': axes_info.get('timepoints', 1),  # Use .get() with default
            'resolutionLevels': len(node.data)
        }





    # Loading Zarr Data
    def loadZarrData(self, url: str, resolutionLevel: int = 0, channel: int = 0, timepoint: int = 0):
        """Main function - keep it simple and delegate"""
        if not self.reader or not self.nodes:
            self.connectToZarr(url)
        
        try:
            print(f"Loading OME-Zarr data: level={resolutionLevel}, channel={channel}, timepoint={timepoint}")
            
            node = self.nodes[0]
            level = min(resolutionLevel, len(node.data) - 1)
            print(f"DEBUG: Adjusted level from {resolutionLevel} to {level} (max available: {len(node.data) - 1})")
            
            # Delegate to smaller functions
            volume = self._extractVolumeData(node, level, channel, timepoint)
            print(f"DEBUG: Extracted volume shape: {volume.shape}")
            
            spacing = self._extractSpacing(node, resolutionLevel)
            print(f"DEBUG: Extracted spacing: {spacing}")
            
            volumeNode = self._createVolumeNode(volume, spacing, resolutionLevel, channel, timepoint)
            
            print(f"OME-Zarr data loaded successfully")
            return volumeNode
            
        except Exception as e:
            print(f"ERROR: Failed to load OME-Zarr data: {e}")
            raise

    def _extractVolumeData(self, node, level, channel, timepoint=0):
        """Extract volume data using actual axis order from metadata"""
        raw_data = node.data[level]
        print(f"DEBUG: Raw data shape: {raw_data.shape}")
        
        if len(raw_data.shape) == 3:
            # Already 3D
            volume = raw_data
            print(f"DEBUG: Data already 3D: {volume.shape}")
        elif len(raw_data.shape) > 3:
            # Multi-dimensional data - use axis order to slice correctly
            volume = self._sliceByAxisOrder(raw_data, channel, timepoint)
        elif len(raw_data.shape) == 2:
            # 2D data - convert to 3D by adding a Z dimension
            volume = raw_data[numpy.newaxis, :, :]  # Shape becomes (1, Y, X)
            print(f"DEBUG: Converted 2D to 3D: {raw_data.shape} → {volume.shape}")
        elif len(raw_data.shape) == 1:
            # 1D data - convert to 3D 
            volume = raw_data[numpy.newaxis, numpy.newaxis, :]  # Shape becomes (1, 1, X)
            print(f"DEBUG: Converted 1D to 3D: {raw_data.shape} → {volume.shape}")
        else:
            raise ValueError(f"Unsupported data dimensionality: {raw_data.shape}")

        if len(volume.shape) != 3:
            raise ValueError(f"Expected 3D volume after extraction, got shape: {volume.shape}")

        return volume


    def _sliceByAxisOrder(self, raw_data, channel, timepoint):
        """Slice multi-dimensional data using stored axis order"""
        if not hasattr(self, 'axis_order') or not self.axis_order:
            print("WARNING: No axis order available, using fallback method")
            return self._fallbackSlicing(raw_data, channel, timepoint)
    
        print(f"DEBUG: Using axis order: {self.axis_order}")
        print(f"DEBUG: Axis names: {self.axis_names}")
    
        # Build indexing tuple based on axis order
        indexing = []
        for i, axis_type in enumerate(self.axis_order):
            if axis_type == "channel":
                indexing.append(channel)
                print(f"DEBUG: Setting channel={channel} at axis {i}")
            elif axis_type == "time":
                indexing.append(timepoint)
                print(f"DEBUG: Setting timepoint={timepoint} at axis {i}")
            elif axis_type == "space":
                indexing.append(slice(None))  # Keep all spatial data
                print(f"DEBUG: Keeping all data for spatial axis {i}")
            else:
                # Unknown axis type, keep first element
                indexing.append(0)
                print(f"DEBUG: Using index 0 for unknown axis type '{axis_type}' at axis {i}")
    
        # Convert to tuple and slice
        indexing_tuple = tuple(indexing)
        print(f"DEBUG: Final indexing tuple: {indexing_tuple}")
    
        volume = raw_data[indexing_tuple]
        print(f"DEBUG: Sliced volume shape: {volume.shape}")
    
        return volume


    def _fallbackSlicing(self, raw_data, channel, timepoint):
        """Fallback slicing when axis order is unknown"""
        print("DEBUG: Using fallback slicing logic")
    
        if len(raw_data.shape) == 4:
            volume = raw_data[channel]
            print(f"DEBUG: 4D fallback - extracted channel {channel}, shape: {volume.shape}")
        elif len(raw_data.shape) == 5:
            # Use the old heuristic
            if hasattr(self, 'time_axis') and hasattr(self, 'channel_axis'):
                if self.time_axis < self.channel_axis:
                    volume = raw_data[timepoint, channel]
                else:
                    volume = raw_data[channel, timepoint]
            else:
                volume = raw_data[timepoint, channel] if raw_data.shape[0] < raw_data.shape[1] else raw_data[channel, timepoint]
            print(f"DEBUG: 5D fallback - extracted timepoint {timepoint}, channel {channel}, shape: {volume.shape}")
        else:
            raise ValueError(f"Unsupported data dimensionality: {raw_data.shape}")
    
        return volume

    def _extractSpacing(self, node, resolutionLevel):
        """Extract spacing information from metadata"""
        spacing = [1.0, 1.0, 1.0]  # Default [x, y, z]
        
        print(f"DEBUG: _extractSpacing called with resolutionLevel: {resolutionLevel}")
        
        try:
            if hasattr(node, 'metadata') and 'coordinateTransformations' in node.metadata:
                print(f"DEBUG: Found coordinateTransformations in metadata")
                print(f"DEBUG: Available resolution levels: {len(node.metadata['coordinateTransformations'])}")
                
                transforms = node.metadata['coordinateTransformations'][resolutionLevel]
                print(f"DEBUG: Selected transforms for level {resolutionLevel}: {transforms}")
                
                for transform in transforms:
                    if transform.get('type') == 'scale':
                        scale = transform.get('scale', [])
                        print(f"DEBUG: Found scale transform: {scale}")
                        
                        spacing = self._mapScaleToSpacing(scale)
                        print(f"DEBUG: Mapped spacing result: {spacing}")
                        break
            else:
                print("DEBUG: No coordinateTransformations found in metadata")
                    
        except Exception as e:
            print(f"ERROR: Could not extract spacing: {e}")
    
        print(f"DEBUG: Final spacing returned: {spacing}")
        return spacing

    def _mapScaleToSpacing(self, scale):
        """Map OME-Zarr scale to VTK spacing using stored axis info"""
        spacing = [1.0, 1.0, 1.0]
        
        print(f"DEBUG: _mapScaleToSpacing called with scale: {scale}")
        print(f"DEBUG: Available axis info - x: {hasattr(self, 'x_axis_info')}, y: {hasattr(self, 'y_axis_info')}, z: {hasattr(self, 'z_axis_info')}")
        
        if hasattr(self, 'x_axis_info') and self.x_axis_info:
            x_index = self.x_axis_info['index']
            print(f"DEBUG: X-axis index: {x_index}")
            if len(scale) > x_index:
                spacing[0] = scale[x_index]
                print(f"DEBUG: Set X spacing to {spacing[0]} from scale[{x_index}]")
        
        if hasattr(self, 'y_axis_info') and self.y_axis_info:
            y_index = self.y_axis_info['index']
            print(f"DEBUG: Y-axis index: {y_index}")
            if len(scale) > y_index:
                spacing[1] = scale[y_index]
                print(f"DEBUG: Set Y spacing to {spacing[1]} from scale[{y_index}]")
        
        if hasattr(self, 'z_axis_info') and self.z_axis_info:
            z_index = self.z_axis_info['index']
            print(f"DEBUG: Z-axis index: {z_index}")
            if len(scale) > z_index:
                spacing[2] = scale[z_index]
                print(f"DEBUG: Set Z spacing to {spacing[2]} from scale[{z_index}]")
        
        print(f"DEBUG: Final mapped spacing: {spacing}")
        return spacing

    def _createVolumeNode(self, volume, spacing, resolutionLevel, channel, timepoint=0):
        """Create and configure VTK volume node"""
        print(f"DEBUG: _createVolumeNode called with spacing: {spacing}")
        
        # VTK setup
        voxelType = vtk.VTK_UNSIGNED_SHORT
        dimensions = list(volume.shape)
        dimensions.reverse()
        
        print(f"DEBUG: Volume shape: {volume.shape}")
        print(f"DEBUG: VTK dimensions (reversed): {dimensions}")

        imageData = vtk.vtkImageData()
        imageData.SetDimensions(dimensions)
        imageData.SetSpacing(1.0, 1.0, 1.0)   # Actual spacing seems to distorts pixel data.
        imageData.SetOrigin(0.0, 0.0, 0.0)
        
        print(f"DEBUG: VTK ImageData configured with:")
        print(f"  - Dimensions: {imageData.GetDimensions()}")
        print(f"  - Spacing: {imageData.GetSpacing()}")
        print(f"  - Origin: {imageData.GetOrigin()}")
        
        imageData.AllocateScalars(voxelType, 1)

        # Create volume node with timepoint info
        volumeName = f"OME-Zarr_L{resolutionLevel}_C{channel}_T{timepoint}"
        volumeNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode", volumeName)
        volumeNode.SetAndObserveImageData(imageData)
        volumeNode.CreateDefaultDisplayNodes()
        volumeNode.CreateDefaultStorageNode()

        volumeNode.SetSpacing(spacing[0], spacing[1], spacing[2])    # Set actual spacing    
        print(f"DEBUG: Volume node '{volumeName}' created with spacing: {volumeNode.GetSpacing()}")

        # Load data
        array = slicer.util.arrayFromVolume(volumeNode)
        array[:] = volume
        slicer.util.arrayFromVolumeModified(volumeNode)
        
        print(f"DEBUG: Volume node '{volumeName}' created successfully")
        return volumeNode


#
# SlicerZarrOMETest
#

'''

class SlicerZarrOMETest(ScriptedLoadableModuleTest):
    """
    This is the test case for your scripted module.
    Uses ScriptedLoadableModuleTest base class, available at:
    https://github.com/Slicer/Slicer/blob/main/Base/Python/slicer/ScriptedLoadableModule.py
    """

    def setUp(self):
        """Do whatever is needed to reset the state - typically a scene clear will be enough."""
        slicer.mrmlScene.Clear()

    def runTest(self):
        """Run as few or as many tests as needed here."""
        self.setUp()
        self.test_SlicerZarrOME1()

    def test_SlicerZarrOME1(self):
        """Ideally you should have several levels of tests.  At the lowest level
        tests should exercise the functionality of the logic with different inputs
        (both valid and invalid).  At higher levels your tests should emulate the
        way the user would interact with your code and confirm that it still works
        the way you intended.
        One of the most important features of the tests is that it should alert other
        developers when their changes will have an impact on the behavior of your
        module.  For example, if a developer removes a feature that you depend on,
        your test should break so they know that the feature is needed.
        """

        self.delayDisplay("Starting the test")

        # Get/create input data

        import SampleData

        registerSampleData()
        inputVolume = SampleData.downloadSample("SlicerZarrOME1")
        self.delayDisplay("Loaded test data set")

        inputScalarRange = inputVolume.GetImageData().GetScalarRange()
        self.assertEqual(inputScalarRange[0], 0)
        self.assertEqual(inputScalarRange[1], 695)

        outputVolume = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLScalarVolumeNode")
        threshold = 100

        # Test the module logic

        logic = SlicerZarrOMELogic()

        # Test algorithm with non-inverted threshold
        logic.process(inputVolume, outputVolume, threshold, True)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], threshold)

        # Test algorithm with inverted threshold
        logic.process(inputVolume, outputVolume, threshold, False)
        outputScalarRange = outputVolume.GetImageData().GetScalarRange()
        self.assertEqual(outputScalarRange[0], inputScalarRange[0])
        self.assertEqual(outputScalarRange[1], inputScalarRange[1])

        self.delayDisplay("Test passed")

        '''