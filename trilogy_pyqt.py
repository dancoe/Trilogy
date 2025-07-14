"""
Trilogy PyQt - Fast interactive color image creation for large astronomical images
https://github.com/dancoe/trilogy
by Dan Coe + AI (Gemini, Claude, Cursor)

This PyQt application provides a native desktop interface for creating stunning color 
composites from astronomical FITS files. It is designed for performance, capable of 
handling massive images (e.g., 17000x10000+ pixels) with real-time feedback.

Quick Start:
    pip install PyQt5 numpy astropy pillow scipy numba
    python trilogy_pyqt.py

Features:
- Native performance for very large images with hardware-accelerated graphics.
- Smooth, interactive zooming and panning with trackpad and mouse support.
- Real-time parameter adjustment with sliders for noise, saturation, and scaling.
- Flexible color modes: Assign colors automatically in RGB or a full rainbow spectrum.
- Customizable filter colors with an interactive color picker.
- Live coordinate display showing pixel (x, y) and World Coordinate System (RA/Dec).
- Save current view ("stamp") or full-resolution image.
- Standalone desktop application or can be integrated with Jupyter notebooks.
- Cross-platform support (tested on macOS, Windows, Linux).

Controls:
- Trackpad:
  - Pinch (two fingers): Zoom in/out.
  - Two-finger scroll: Pan within the zoomed image.
  - Single-finger drag: Move the sample position for scaling calculation.
  - Double-click: Recenter the view on the clicked point.
- Mouse:
  - Scroll wheel: Zoom in/out.
  - Drag: Move the sample position.
  - Double-click: Recenter the view.
- General:
  - Load Files: Select one or more FITS files to begin.
  - Filter Panel: Select/deselect filters, change their assigned colors.
  - Parameter Panel: Adjust image scaling and color saturation in real-time.
  - Save Stamp: Save the current zoomed-in view.
  - Save Full Image: Save the entire image with current settings.
"""

import sys
import os
import numpy as np
from glob import glob
import warnings
from copy import deepcopy
import json
import time
import re
warnings.filterwarnings('ignore')

# PyQt imports - try both PyQt5 and PySide2
try:
    from PyQt5.QtWidgets import *
    from PyQt5.QtCore import *
    from PyQt5.QtGui import *
    print("Using PyQt5")
    QT_VERSION = "PyQt5"
except ImportError:
    try:
        from PySide2.QtWidgets import *
        from PySide2.QtCore import *
        from PySide2.QtGui import *
        print("Using PySide2")
        QT_VERSION = "PySide2"
    except ImportError:
        print("ERROR: Neither PyQt5 nor PySide2 found!")
        print("Install with: pip install PyQt5  or  pip install PySide2")
        sys.exit(1)

# Core libraries
import astropy.io.fits as fits
from astropy.stats import sigma_clipped_stats
from astropy.visualization import simple_norm
from astropy.coordinates import SkyCoord
from astropy import units as u

# Image processing
from PIL import Image, ImageEnhance
import PIL
PIL.Image.MAX_IMAGE_PIXELS = 933120000

# Optimization
from numba import jit
from scipy.optimize import golden

# For PNG metadata
import matplotlib.colors
from PIL.PngImagePlugin import PngInfo


# =============================================================================
# CORE TRILOGY FUNCTIONS (same as optimized version)
# =============================================================================

@jit(nopython=True)
def fast_imscale(data, x0, x1, x2, y1, k):
    """Optimized image scaling function"""
    r1 = np.log10(k * (x2 - x0) + 1)
    v = data.ravel()
    v = np.clip(v, 0, None)
    d = k * (v - x0) + 1
    d = np.clip(d, 1e-30, None)
    z = np.log10(d) / r1
    z = np.clip(z, 0, 1)
    z = z.reshape(data.shape)
    return (z * 255).astype(np.uint8)

def solve_k(x0, x1, x2, y1):
    """Solve for k parameter in tri-log scaling"""
    if y1 == 0.5:
        return (x2 - 2 * x1 + x0) / float(x1 - x0) ** 2
    else:
        n = 1 / y1
        def da(k):
            a1 = k * (x1 - x0) + 1
            a2 = k * (x2 - x0) + 1
            a1n = np.abs(a1**n)
            da1 = a1n - a2
            k = np.abs(k)
            if k == 0:
                return 1e10
            else:
                da1 = da1 / k
            return abs(da1)
        return np.abs(golden(da))

def extract_filter_names_from_files(file_list):
    """
    Extracts filter names by finding which part of the filenames is different.
    Assumes files share a common naming scheme with one varying component.
    Returns a list of tuples (filename, filter_name).
    """
    if not file_list:
        return []
    if len(file_list) == 1:
        # Fallback for single file
        return [(file_list[0], extract_filter_name_legacy(file_list[0]))]

    basenames = [os.path.basename(f) for f in file_list]
    
    def tokenize(s):
        name_without_ext, _ = os.path.splitext(s)
        return re.split('[-_]', name_without_ext)

    tokenized_names = [tokenize(b) for b in basenames]
    
    # Check for consistent filename structure
    if not all(len(t) == len(tokenized_names[0]) for t in tokenized_names):
         print("⚠️ Filename structures differ, falling back to legacy filter extraction.")
         return [(f, extract_filter_name_legacy(f)) for f in file_list]
    
    varying_indices = [
        i for i in range(len(tokenized_names[0]))
        if len({tokens[i] for tokens in tokenized_names}) > 1
    ]

    if len(varying_indices) != 1:
        print(f"⚠️ Found {len(varying_indices)} varying parts in filenames, expected 1. Falling back to legacy filter extraction.")
        return [(f, extract_filter_name_legacy(f)) for f in file_list]

    varying_idx = varying_indices[0]
    
    filters = []
    for i, f in enumerate(file_list):
        filter_name = tokenized_names[i][varying_idx]
        filters.append((f, filter_name))
    
    return filters

def extract_filter_name_legacy(filename):
    """Extract filter name from filename (legacy fallback)"""
    return os.path.basename(filename).split('_')[0].lower().split('-')[1]

def normalize_array_dtype(arr):
    """Convert array to native byte order and ensure proper dtype"""
    if arr.dtype.byteorder not in ('=', '|'):
        arr = arr.astype(arr.dtype.newbyteorder('='))
    return np.ascontiguousarray(arr, dtype=np.float32)

def fast_determine_scaling(data, unsatpercent, noisesig=1, correctbias=True, noisefloorsig=2):
    """Fast scaling determination"""
    data = normalize_array_dtype(data)
    valid_data = data[~np.isnan(data)]
    
    if len(valid_data) == 0:
        return 0, 1, 100
    
    sorted_data = np.sort(valid_data.flatten())
    
    if sorted_data[0] == sorted_data[-1]:
        return 0, 1, 100
    
    mean, median, stddev = sigma_clipped_stats(sorted_data)
    
    if correctbias:
        x0 = mean - noisefloorsig * stddev
    else:
        x0 = 0
    
    x1 = mean + noisesig * stddev
    
    sat_idx = int((unsatpercent / 100.0) * len(sorted_data))
    sat_idx = np.clip(sat_idx, 0, len(sorted_data) - 1)
    x2 = sorted_data[sat_idx]
    
    return x0, x1, x2


# =============================================================================
# PYQT CUSTOM WIDGETS
# =============================================================================

class ImageView(QGraphicsView):
    """High-performance image viewer with position dragging"""
    
    positionChanged = pyqtSignal(int, int) if QT_VERSION == "PyQt5" else Signal(int, int)
    coordinatesChanged = pyqtSignal(int, int, str, str) if QT_VERSION == "PyQt5" else Signal(int, int, str, str)
    zoomChanged = pyqtSignal(float) if QT_VERSION == "PyQt5" else Signal(float)
    centerChanged = pyqtSignal(int, int) if QT_VERSION == "PyQt5" else Signal(int, int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Setup graphics scene
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        # Setup view properties - DISABLE rubber band selection
        self.setDragMode(QGraphicsView.NoDrag)  # No rubber band
        self.setRenderHint(QPainter.Antialiasing, False)  # Faster for large images
        self.setRenderHint(QPainter.SmoothPixmapTransform, False)
        
        # Disable scrollbars
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # Enable mouse tracking for coordinate display
        self.setMouseTracking(True)
        
        # Enable gesture support for trackpad
        self.setAttribute(Qt.WA_AcceptTouchEvents)
        self.grabGesture(Qt.PinchGesture)
        self.grabGesture(Qt.PanGesture)
        
        # Image item
        self.image_item = None
        self.current_pixmap = None
        
        # Position dragging
        self.dragging = False
        self.last_pos = QPoint()
        self.image_bounds = None
        
        # Min/Max zoom levels
        self.min_zoom_level = 1.0
        self.max_zoom_level = 50.0  # e.g., 1 data pixel = 50 screen pixels

        # Coordinate tracking
        self.wcs_info = None  # For RA/Dec conversion
        self.stamp_offset = (0, 0)  # Offset of current stamp in full image
        self.full_image_size = None  # Size of full image
        
        # Gesture state
        self.pinch_in_progress = False
        self.pan_in_progress = False
        
    def setWCSInfo(self, wcs_info):
        """Set WCS information for RA/Dec conversion"""
        self.wcs_info = wcs_info
    
    def setStampOffset(self, offset_x, offset_y, full_width, full_height):
        """Set the offset of the current stamp within the full image"""
        self.stamp_offset = (offset_x, offset_y)
        self.full_image_size = (full_width, full_height)
        
    def setImage(self, image_array):
        """Set image from numpy array with Y-axis flip"""
        if image_array is None:
            if self.image_item:
                self.scene.removeItem(self.image_item)
                self.image_item = None
            return
            
        # Convert numpy array to QImage
        if len(image_array.shape) == 3:  # Color image
            height, width, channels = image_array.shape
            bytes_per_line = channels * width
            
            if channels == 3:
                # Use tobytes() to fix the conversion issue
                qimage = QImage(image_array.tobytes(), width, height, bytes_per_line, QImage.Format_RGB888)
            elif channels == 4:
                qimage = QImage(image_array.tobytes(), width, height, bytes_per_line, QImage.Format_RGBA8888)
        else:  # Grayscale
            height, width = image_array.shape
            bytes_per_line = width
            qimage = QImage(image_array.tobytes(), width, height, bytes_per_line, QImage.Format_Grayscale8)
        
        # **FIX: Apply Y-axis flip to correct astronomical image orientation**
        qimage = qimage.mirrored(False, True)  # Flip vertically
        
        # Convert to pixmap
        self.current_pixmap = QPixmap.fromImage(qimage)
        
        # Store image bounds for dragging
        self.image_bounds = (width, height)
        
        # If image item exists, just update its pixmap. Otherwise, create it.
        if self.image_item:
            self.image_item.setPixmap(self.current_pixmap)
        else:
            self.scene.clear()
            self.image_item = self.scene.addPixmap(self.current_pixmap)
        
    def mousePressEvent(self, event):
        """Start position dragging"""
        if event.button() == Qt.LeftButton and self.image_item:
            self.dragging = True
            self.last_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)
                
        super().mousePressEvent(event)
    
    def wheelEvent(self, event):
        """Simple mouse wheel zoom with center-based zooming"""
        # Don't handle wheel events if trackpad gestures are active
        if self.pinch_in_progress or self.pan_in_progress:
            return
        
        # Smaller zoom steps for more granular control
        factor = 1.1  # Reduced from 1.2 for smoother zooming
        if event.angleDelta().y() < 0:
            factor = 1.0 / factor
        
        current_zoom = self.transform().m11()
        
        # Enforce max zoom
        if factor > 1.0 and current_zoom * factor > self.max_zoom_level:
            if current_zoom < self.max_zoom_level:
                 scale_to_max = self.max_zoom_level / current_zoom
                 self.scale(scale_to_max, scale_to_max)
                 self.zoomChanged.emit(self.transform().m11() / self.min_zoom_level)
            return

        # Enforce min zoom - call reset for a clean state
        if factor < 1.0 and current_zoom * factor < self.min_zoom_level:
            if abs(current_zoom - self.min_zoom_level) > 1e-6:
                self.resetZoom()
            return

        self.scale(factor, factor)
        self.zoomChanged.emit(self.transform().m11() / self.min_zoom_level)
    
    def mouseMoveEvent(self, event):
        """Handle position dragging and coordinate display"""
        if self.image_item and self.image_bounds:
            # Get image coordinates relative to stamp
            scene_pos = self.mapToScene(event.pos())
            stamp_x = int(scene_pos.x())
            stamp_y = int(scene_pos.y())
            
            # Calculate full image X coordinate
            offset_x, offset_y = self.stamp_offset
            full_img_x = stamp_x + offset_x
            
            # Convert Y coordinate back (account for flip)
            stamp_width, stamp_height = self.image_bounds
            actual_stamp_y = stamp_height - 1 - stamp_y
            
            # Calculate full image Y coordinate
            if self.full_image_size:
                _, full_height = self.full_image_size
                full_img_y = actual_stamp_y + offset_y
            else:
                full_img_y = actual_stamp_y

            # Calculate RA/Dec regardless of whether cursor is inside stamp bounds
            ra_str, dec_str = self.pixelToRADec(full_img_x, full_img_y)
            
            # Emit full image coordinates for display
            self.coordinatesChanged.emit(full_img_x, full_img_y, ra_str, dec_str)
            
            # Unify dragging to always update the center position (xc, yc)
            if self.dragging:
                current_pos = event.pos()
                delta = current_pos - self.last_pos
                self.last_pos = current_pos
                
                # Convert screen delta to image coordinates, respecting current zoom
                scene_p1 = self.mapToScene(0, 0)
                scene_p2 = self.mapToScene(delta.x(), delta.y())
                scene_delta = scene_p2 - scene_p1
                
                # Fix dragging direction
                dx = int(-scene_delta.x())
                dy = int(scene_delta.y()) # Y is flipped in display, so this is correct
                
                self.positionChanged.emit(dx, dy)
        
        super().mouseMoveEvent(event)
    
    def mouseDoubleClickEvent(self, event):
        """Recenter the view on double-click"""
        if event.button() == Qt.LeftButton and self.image_item:
            # Get scene coordinates from the click - this works for any zoom level
            scene_pos = self.mapToScene(event.pos())
            stamp_x = int(scene_pos.x())
            stamp_y = int(scene_pos.y())
            
            # Ensure coordinates are within the stamp bounds
            if self.image_bounds:
                stamp_width, stamp_height = self.image_bounds
                if not (0 <= stamp_x < stamp_width and 0 <= stamp_y < stamp_height):
                    return # Click was outside the image area
                
                # Convert Y coordinate back (account for flip)
                actual_stamp_y = stamp_height - 1 - stamp_y
                
                # Convert stamp coordinates to full image coordinates
                offset_x, offset_y = self.stamp_offset
                full_img_x = stamp_x + offset_x
                full_img_y = actual_stamp_y + offset_y
                
                print(f"Double-click at stamp ({stamp_x}, {stamp_y}) -> full image ({full_img_x}, {full_img_y})")
                
                # Emit signal with new center coordinates
                self.centerChanged.emit(full_img_x, full_img_y)

        super().mouseDoubleClickEvent(event)
    
    def mouseReleaseEvent(self, event):
        """Stop position dragging"""
        if event.button() == Qt.LeftButton and self.dragging:
            self.dragging = False
            self.setCursor(Qt.ArrowCursor)
                
        super().mouseReleaseEvent(event)
    
    def resetZoom(self):
        """Reset zoom to fit image and define this as the base zoom level."""
        if self.image_item:
            self.fitInView(self.image_item, Qt.KeepAspectRatio)
            self.min_zoom_level = self.transform().m11() # Recalculate base zoom
            self.zoomChanged.emit(1.0) # Emit relative zoom of 1.0x
    
    def setZoomLevel(self, zoom_level):
        """Set zoom level (1.0 = fit to view)"""
        if self.image_item:
            self.fitInView(self.image_item, Qt.KeepAspectRatio)
            self.scale(zoom_level, zoom_level)
            if scale_factor > 1.0 and current_zoom * scale_factor > self.max_zoom_level:
                if current_zoom < self.max_zoom_level:
                    scale_to_max = self.max_zoom_level / current_zoom
                    # Center the zoom on the image center
                    if self.image_item:
                        center_point = self.image_item.boundingRect().center()
                        self.centerOn(center_point)
                        self.scale(scale_to_max, scale_to_max)
                    self.zoomChanged.emit(self.transform().m11() / self.min_zoom_level)
                return

            # Check against min zoom - stop smoothly
            if scale_factor < 1.0 and current_zoom * scale_factor < self.min_zoom_level:
                if abs(current_zoom - self.min_zoom_level) > 1e-6:
                    # Directly scale to minimum to avoid 'bouncing'
                    scale_to_min = self.min_zoom_level / current_zoom
                    # Center the zoom on the image center
                    if self.image_item:
                        center_point = self.image_item.boundingRect().center()
                        self.centerOn(center_point)
                        self.scale(scale_to_min, scale_to_min)
                    self.zoomChanged.emit(1.0)
                return

            # Apply zoom - center on image center
            if scale_factor > 0:
                if self.image_item:
                    center_point = self.image_item.boundingRect().center()
                    self.centerOn(center_point)
                    self.scale(scale_factor, scale_factor)
                self.zoomChanged.emit(self.transform().m11() / self.min_zoom_level)
    
    def saveViewState(self):
        """Save current view state (zoom and scroll position)"""
        transform = self.transform()
        scroll_x = self.horizontalScrollBar().value()
        scroll_y = self.verticalScrollBar().value()
        return {
            'transform': transform,
            'scroll_x': scroll_x,
            'scroll_y': scroll_y,
            'zoom_factor': transform.m11()  # Get current zoom factor
        }
    
    def restoreViewState(self, view_state):
        """Restore view state (zoom and scroll position)"""
        if view_state and self.image_item:
            try:
                # Restore transform (zoom)
                self.setTransform(view_state['transform'])
                
                # Restore scroll position
                self.horizontalScrollBar().setValue(view_state['scroll_x'])
                self.verticalScrollBar().setValue(view_state['scroll_y'])
            except:
                # If restore fails, just fit to view
                self.fitInView(self.image_item, Qt.KeepAspectRatio)

    def pixelToRADec(self, x, y):
        """Convert pixel coordinates to RA/Dec strings"""
        if self.wcs_info:
            try:
                # Use WCS to convert pixel to world coordinates
                ra, dec = self.wcs_info.pixel_to_world_values(x, y)
                
                # Create SkyCoord object for formatting
                coords = SkyCoord(ra, dec, unit='deg', frame='icrs')
                
                # Format to HMS/DMS and decimal
                ra_hms_str = coords.ra.to_string(unit=u.hourangle, sep=':', pad=True, precision=2)
                dec_dms_str = coords.dec.to_string(unit=u.degree, sep=':', alwayssign=True, pad=True, precision=1)
                
                # Combine into final strings
                ra_str = f"RA: {ra_hms_str} = {ra:.6f}°"
                dec_str = f"Dec: {dec_dms_str} = {dec:.6f}°"

                return ra_str, dec_str
            except:
                pass
        
        # Fallback: show placeholder
        return "RA: N/A", "Dec: N/A"

    def event(self, event):
        """Handle gesture events for trackpad"""
        if event.type() == QEvent.Gesture:
            return self.gestureEvent(event)
        return super().event(event)
    
    def gestureEvent(self, event):
        """Handle trackpad gestures"""
        try:
            pinch = event.gesture(Qt.PinchGesture)
            pan = event.gesture(Qt.PanGesture)
            
            if pinch:
                self.pinchGesture(pinch)
            if pan:
                self.panGesture(pan)
                
            return True
        except:
            return False
    
    def pinchGesture(self, gesture):
        """Handle pinch gesture for zooming"""
        if gesture.state() == Qt.GestureStarted:
            self.pinch_in_progress = True
        elif gesture.state() == Qt.GestureFinished:
            self.pinch_in_progress = False
        
        if gesture.state() in [Qt.GestureStarted, Qt.GestureUpdated]:
            # Get scale factor
            scale_factor = gesture.scaleFactor()
            
            # Check against max zoom
            current_zoom = self.transform().m11()
            if scale_factor > 1.0 and current_zoom * scale_factor > self.max_zoom_level:
                if current_zoom < self.max_zoom_level:
                    scale_to_max = self.max_zoom_level / current_zoom
                    self.scale(scale_to_max, scale_to_max)
                return

            # Check against min zoom
            if scale_factor < 1.0 and current_zoom * scale_factor < self.min_zoom_level:
                if current_zoom > self.min_zoom_level:
                    scale_to_min = self.min_zoom_level / current_zoom
                    self.scale(scale_to_min, scale_to_min)
                return

            # Apply zoom
            if scale_factor > 0:
                self.scale(scale_factor, scale_factor)
                # Temporarily disable zoom level tracking for performance
                # new_zoom = self.transform().m11()
                # self.zoomChanged.emit(new_zoom)
    
    def panGesture(self, gesture):
        """Handle pan gesture for two-finger scrolling"""
        if gesture.state() == Qt.GestureStarted:
            self.pan_in_progress = True
        elif gesture.state() == Qt.GestureFinished:
            self.pan_in_progress = False
        
        if gesture.state() in [Qt.GestureStarted, Qt.GestureUpdated]:
            # Get pan delta
            delta = gesture.delta()
            
            # Apply scroll offset
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x())
            )
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(delta.y())
            )


class FilterPanel(QWidget):
    """Panel for filter selection with color pickers"""
    
    filterChanged = pyqtSignal() if QT_VERSION == "PyQt5" else Signal()
    colorChanged = pyqtSignal(str, tuple) if QT_VERSION == "PyQt5" else Signal(str, tuple)
    autoSetColors = pyqtSignal() if QT_VERSION == "PyQt5" else Signal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.filter_checkboxes = {}
        self.color_buttons = {}
        self.filter_colors = {}
        self.color_mode = "rainbow"  # "rainbow" or "rgb"
        self.purple_checkbox = None
        
        self.initUI()
    
    def initUI(self):
        """Initialize the filter panel UI"""
        layout = QVBoxLayout(self)
        
        # Title
        title_label = QLabel("Filters")
        title_label.setStyleSheet("font-weight: bold; font-size: 14px;")
        layout.addWidget(title_label)
        
        # Direct layout for filters (no scroll area)
        self.filter_layout = QVBoxLayout()
        layout.addLayout(self.filter_layout)
        
        # Control buttons - make them wider
        button_layout = QHBoxLayout()
        
        select_all_btn = QPushButton("All")
        select_all_btn.setMinimumWidth(60)  # Make wider
        select_all_btn.clicked.connect(self.selectAllFilters)
        
        select_none_btn = QPushButton("None")
        select_none_btn.setMinimumWidth(60)  # Make wider
        select_none_btn.clicked.connect(self.selectNoFilters)
        
        button_layout.addWidget(select_all_btn)
        button_layout.addWidget(select_none_btn)
        
        layout.addLayout(button_layout)
        
        # Color mode selector
        mode_layout = QVBoxLayout()
        mode_label = QLabel("Color Selection Mode:")
        mode_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        mode_layout.addWidget(mode_label)
        
        # Radio buttons for mode selection - REORDERED
        self.rgb_radio = QRadioButton("RGB")
        self.rainbow_radio = QRadioButton("Rainbow")
        
        self.rainbow_radio.toggled.connect(self.onModeChanged)
        
        mode_layout.addWidget(self.rgb_radio)
        mode_layout.addWidget(self.rainbow_radio)

        # Purple checkbox for rainbow mode
        self.purple_checkbox = QCheckBox("Purple")
        self.purple_checkbox.setChecked(False) # Default to not include purple
        self.purple_checkbox.setStyleSheet("margin-left: 20px;") # Indent it
        # self.purple_checkbox.stateChanged.connect(self.autoSetColors.emit) # *** REMOVED: No longer auto-triggers
        mode_layout.addWidget(self.purple_checkbox)
        
        layout.addLayout(mode_layout)

        # Default to rainbow mode, which also triggers onModeChanged to set visibility
        self.rainbow_radio.setChecked(True)
        
        # Auto color button
        auto_color_btn = QPushButton("Auto Set Colors")
        auto_color_btn.setToolTip("Automatically assign colors to all filters")
        auto_color_btn.clicked.connect(self.autoSetColors.emit)
        layout.addWidget(auto_color_btn)
        
        layout.addStretch()
        
        # Set maximum width to keep it compact
        self.setMaximumWidth(180)
    
    def onModeChanged(self):
        """Handle color mode change and purple checkbox visibility"""
        is_rainbow = self.rainbow_radio.isChecked()
        
        if is_rainbow:
            self.color_mode = "rainbow"
            self.purple_checkbox.setVisible(True)
        else:
            self.color_mode = "rgb"
            self.purple_checkbox.setVisible(False)
        
        print(f"Color mode changed to: {self.color_mode}")
        
        # DO NOT automatically update colors when mode changes
        # self.autoSetColors.emit()
    
    def addFilter(self, filter_name, color):
        """Add a filter with checkbox and color picker"""
        # Container for filter row
        filter_widget = QWidget()
        filter_layout = QHBoxLayout(filter_widget)
        filter_layout.setContentsMargins(0, 0, 0, 0)
        
        # Checkbox
        checkbox = QCheckBox(filter_name.upper())
        checkbox.setChecked(True)
        checkbox.stateChanged.connect(self.filterChanged.emit)
        self.filter_checkboxes[filter_name] = checkbox
        
        # Color button
        color_btn = QPushButton()
        color_btn.setMaximumSize(30, 20)
        color_btn.setMinimumSize(30, 20)
        color_btn.clicked.connect(lambda: self.changeColor(filter_name))
        color_btn.setToolTip(f"Click to change {filter_name.upper()} color")
        
        # Set button color
        self.filter_colors[filter_name] = color
        self.color_buttons[filter_name] = color_btn  # Store button reference first
        self.updateColorButton(filter_name, color)   # Then update its appearance
        
        # Layout
        filter_layout.addWidget(checkbox)
        filter_layout.addStretch()
        filter_layout.addWidget(color_btn)
        
        # Add to main layout
        self.filter_layout.addWidget(filter_widget)
    
    def updateColorButton(self, filter_name, color):
        """Update color button appearance"""
        if filter_name in self.color_buttons:
            r, g, b = [int(c * 255) for c in color[:3]]
            self.color_buttons[filter_name].setStyleSheet(
                f"QPushButton {{ background-color: rgb({r}, {g}, {b}); border: 2px solid #333; border-radius: 3px; }} "
                f"QPushButton:pressed {{ border: 2px solid #000; }}"
            )
    
    def changeColor(self, filter_name):
        """Open color picker dialog"""
        if self.color_mode == "rgb":
            # RGB mode: show custom dialog with only R, G, B options
            self.showRGBColorDialog(filter_name)
        else:
            # Rainbow mode: show standard color picker with RGB sliders
            self.showStandardColorDialog(filter_name)
    
    def showRGBColorDialog(self, filter_name):
        """Show inline RGB color selection boxes"""
        # Create a popup widget for RGB selection
        popup = QWidget()
        popup.setWindowFlags(Qt.Popup)
        popup.setWindowTitle(f"RGB Color for {filter_name.upper()}")
        
        layout = QVBoxLayout(popup)  # Changed to vertical layout
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)
        
        # RGB color options as clickable boxes - reordered: Blue, Green, Red
        rgb_colors = [
            ("Blue", (0.0, 0.0, 1.0)),
            ("Green", (0.0, 1.0, 0.0)),
            ("Red", (1.0, 0.0, 0.0))
        ]
        
        for name, color in rgb_colors:
            color_box = QPushButton()  # No text, just color
            color_box.setMinimumSize(40, 30)  # Smaller, more square
            color_box.setMaximumSize(40, 30)
            
            # Set button color
            r, g, b = [int(c * 255) for c in color]
            color_box.setStyleSheet(
                f"QPushButton {{ "
                f"background-color: rgb({r}, {g}, {b}); "
                f"border: 2px solid #333; "
                f"border-radius: 3px; "
                f"}} "
                f"QPushButton:hover {{ "
                f"border: 3px solid #000; "
                f"}} "
                f"QPushButton:pressed {{ "
                f"border: 3px solid #FFF; "
                f"}}"
            )
            
            # Set tooltip to show color name
            color_box.setToolTip(name)
            
            # Connect click to color selection
            color_box.clicked.connect(lambda checked, c=color: self.selectRGBColor(filter_name, c, popup))
            
            layout.addWidget(color_box)
        
        # Position popup near the filter button
        if filter_name in self.color_buttons:
            button_pos = self.color_buttons[filter_name].mapToGlobal(self.color_buttons[filter_name].rect().bottomLeft())
            popup.move(button_pos)
        
        popup.show()
    
    def selectRGBColor(self, filter_name, color, popup):
        """Select an RGB color and close popup"""
        self.filter_colors[filter_name] = color
        self.updateColorButton(filter_name, color)
        self.colorChanged.emit(filter_name, color)
        popup.close()
        print(f"RGB color selected for {filter_name}: {color}")
    
    def showStandardColorDialog(self, filter_name):
        """Show standard color picker with RGB sliders"""
        try:
            current_color = self.filter_colors[filter_name]
            r, g, b = [int(c * 255) for c in current_color[:3]]
            initial_color = QColor(r, g, b)
            
            # Try different approaches based on platform
            import platform
            system = platform.system()
            
            if system == "Darwin":  # macOS
                # On macOS, try native dialog first, then fallback to non-native
                try:
                    # Try native dialog first
                    color = QColorDialog.getColor(initial_color, self, 
                                                f"Choose color for {filter_name.upper()}")
                    if color.isValid():
                        new_color = (color.red()/255.0, color.green()/255.0, color.blue()/255.0)
                        self.filter_colors[filter_name] = new_color
                        self.updateColorButton(filter_name, new_color)
                        self.colorChanged.emit(filter_name, new_color)
                        print(f"Color changed for {filter_name}: {new_color}")
                    else:
                        print(f"Color selection cancelled for {filter_name}")
                    return  # Don't show fallback regardless of cancel/accept
                except:
                    pass  # Fall through to non-native dialog
            
            # Create color dialog with safe options
            color_dialog = QColorDialog(initial_color, self)
            color_dialog.setWindowTitle(f"Choose color for {filter_name.upper()}")
            
            # Use minimal options to avoid compatibility issues
            color_dialog.setOptions(QColorDialog.DontUseNativeDialog)
            
            # Don't force RGB mode - let the dialog use its default
            # color_dialog.setCurrentColorMode(QColorDialog.Rgb)  # This can cause crashes
            
            if color_dialog.exec_() == QColorDialog.Accepted:
                color = color_dialog.selectedColor()
                if color.isValid():
                    # Convert to normalized RGB
                    new_color = (color.red()/255.0, color.green()/255.0, color.blue()/255.0)
                    self.filter_colors[filter_name] = new_color
                    self.updateColorButton(filter_name, new_color)
                    
                    # Emit color change signal
                    self.colorChanged.emit(filter_name, new_color)
                    print(f"Color changed for {filter_name}: {new_color}")
            else:
                print(f"Color selection cancelled for {filter_name}")
                # Don't show fallback on cancel - user explicitly cancelled
                    
        except Exception as e:
            print(f"❌ Error opening color dialog for {filter_name}: {e}")
            # Only show fallback on actual errors, not cancellations
            # Check if this is really an error or just a cancellation
            print(f"⚠️  Color dialog failed for {filter_name}, but not showing fallback as requested")
    
    def showSimpleColorFallback(self, filter_name):
        """Show simple color selection as fallback when standard dialog fails"""
        # Create a simple popup with preset colors
        popup = QWidget()
        popup.setWindowFlags(Qt.Popup)
        popup.setWindowTitle(f"Choose color for {filter_name.upper()}")
        
        layout = QGridLayout(popup)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(5)
        
        # Preset rainbow colors
        preset_colors = [
            ("Red", (1.0, 0.0, 0.0)),
            ("Orange", (1.0, 0.5, 0.0)),
            ("Yellow", (1.0, 1.0, 0.0)),
            ("Green", (0.0, 1.0, 0.0)),
            ("Cyan", (0.0, 1.0, 1.0)),
            ("Blue", (0.0, 0.0, 1.0)),
            ("Purple", (0.5, 0.0, 1.0)),
            ("Magenta", (1.0, 0.0, 1.0)),
            ("White", (1.0, 1.0, 1.0))
        ]
        
        # Create color buttons in a 3x3 grid
        for i, (name, color) in enumerate(preset_colors):
            color_box = QPushButton()
            color_box.setMinimumSize(40, 30)
            color_box.setMaximumSize(40, 30)
            
            # Set button color
            r, g, b = [int(c * 255) for c in color]
            color_box.setStyleSheet(
                f"QPushButton {{ "
                f"background-color: rgb({r}, {g}, {b}); "
                f"border: 2px solid #333; "
                f"border-radius: 3px; "
                f"}} "
                f"QPushButton:hover {{ "
                f"border: 3px solid #000; "
                f"}} "
                f"QPushButton:pressed {{ "
                f"border: 3px solid #FFF; "
                f"}}"
            )
            
            # Set tooltip
            color_box.setToolTip(name)
            
            # Connect click to color selection
            color_box.clicked.connect(lambda checked, c=color: self.selectFallbackColor(filter_name, c, popup))
            
            # Add to grid layout
            row = i // 3
            col = i % 3
            layout.addWidget(color_box, row, col)
        
        # Position popup near the filter button
        if filter_name in self.color_buttons:
            button_pos = self.color_buttons[filter_name].mapToGlobal(self.color_buttons[filter_name].rect().bottomLeft())
            popup.move(button_pos)
        
        popup.show()
    
    def selectFallbackColor(self, filter_name, color, popup):
        """Select a fallback color and close popup"""
        self.filter_colors[filter_name] = color
        self.updateColorButton(filter_name, color)
        self.colorChanged.emit(filter_name, color)
        popup.close()
        print(f"Fallback color selected for {filter_name}: {color}")
    
    def selectAllFilters(self):
        """Select all filters"""
        for checkbox in self.filter_checkboxes.values():
            checkbox.setChecked(True)
    
    def selectNoFilters(self):
        """Deselect all filters"""
        for checkbox in self.filter_checkboxes.values():
            checkbox.setChecked(False)
    
    def isFilterChecked(self, filter_name):
        """Check if a filter is selected"""
        checkbox = self.filter_checkboxes.get(filter_name)
        if checkbox:
            return checkbox.isChecked()
        return False
    
    def getFilterColor(self, filter_name):
        """Get filter color"""
        return self.filter_colors.get(filter_name, (1.0, 1.0, 1.0))
    
    def updateAllColorButtons(self):
        """Update all color buttons to show current colors"""
        for filter_name in self.filter_colors:
            if filter_name in self.color_buttons:
                self.updateColorButton(filter_name, self.filter_colors[filter_name])
    
    def clearFilters(self):
        """Clear all filters"""
        for i in reversed(range(self.filter_layout.count())):
            child = self.filter_layout.itemAt(i).widget()
            if child:
                child.deleteLater()
        
        self.filter_checkboxes.clear()
        self.color_buttons.clear()
        self.filter_colors.clear()


class ParameterPanel(QWidget):
    """Panel with sliders for real-time parameter adjustment"""
    
    parametersChanged = pyqtSignal() if QT_VERSION == "PyQt5" else Signal()
    centerRequest = pyqtSignal(dict) if QT_VERSION == "PyQt5" else Signal(dict)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.params = {
            'noiselum': 0.15,
            'satpercent': 0.01,
            'noisesig': 1.0,
            'noisefloorsig': 2.0,
            'color_saturation': 2.0,
            'sample_size': 1000,
            # xc and yc are removed from here, will be handled by coordinate fields
        }
        
        self.param_configs = {}
        self.sliders = {}
        self.text_inputs = {}
        self.save_btn = None
        self.zoom_reset_btn = None # For new zoom control
        
        self.initUI()
        
    def initUI(self):
        layout = QVBoxLayout(self)

        # Top Button Row: Load Files, Save Stamp, Save Full Image
        button_layout_top = QHBoxLayout()
        load_files_btn = QPushButton("Load Files")
        load_files_btn.setMinimumWidth(92)
        load_files_btn.setToolTip("Select FITS files to load (Shift+Click for multiple)")

        save_stamp_btn = QPushButton("Save Stamp")
        save_stamp_btn.setMinimumWidth(100)

        save_full_btn = QPushButton("Save Full Image")
        save_full_btn.setMinimumWidth(120)

        button_layout_top.addWidget(load_files_btn)
        button_layout_top.addWidget(save_stamp_btn)
        button_layout_top.addWidget(save_full_btn)
        
        layout.addLayout(button_layout_top)
        
        # Create parameter controls
        slider_configs = [
            ('noiselum', 'Noise Luminosity', 0.01, 0.5, 0.01, 100, 3),
            ('satpercent', 'Saturation %', 0.001, 1.0, 0.001, 1000, 3),
            ('noisesig', 'Noise Sigma', 0.1, 5.0, 0.1, 10, 1),
            ('noisefloorsig', 'Noise Floor Sigma', 0.0, 5.0, 0.1, 10, 1),
            ('color_saturation', 'Color Saturation', 0.1, 5.0, 0.1, 10, 1),
            ('sample_size', 'Sample Size', 100, 2000, 50, 1, 0),
            # ('xc', 'X Center', 0, 1000, 10, 1, 0),  -- REMOVED
            # ('yc', 'Y Center', 0, 1000, 10, 1, 0),  -- REMOVED
        ]
        
        for param, label, min_val, max_val, step, scale, decimals in slider_configs:
            # Store config for later use
            self.param_configs[param] = {
                'min': min_val, 'max': max_val, 'step': step, 
                'scale': scale, 'decimals': decimals
            }
            
            # Create a simple widget for each parameter
            param_widget = QWidget()
            param_layout = QVBoxLayout(param_widget)
            param_layout.setContentsMargins(5, 5, 5, 5)
            
            # Top row: label and input controls
            top_row = QHBoxLayout()
            
            param_label = QLabel(f"{label}:")
            
            # Text input field also serves as value display
            text_input = QLineEdit()
            text_input.setMinimumWidth(80)
            
            if decimals == 0:
                initial_text = f"{int(self.params[param])}"
            else:
                initial_text = f"{self.params[param]:.{decimals}f}"
            text_input.setText(initial_text)
            
            text_input.returnPressed.connect(lambda p=param: self.updateFromText(p))
            self.text_inputs[param] = text_input
            
            # Increment/decrement buttons
            btn_layout = QVBoxLayout()
            btn_layout.setSpacing(0)
            
            up_btn = QPushButton("▲")
            up_btn.setMaximumSize(30, 20)
            up_btn.clicked.connect(lambda checked, p=param, s=step: self.incrementParameter(p, s))
            
            down_btn = QPushButton("▼")
            down_btn.setMaximumSize(30, 20)
            down_btn.clicked.connect(lambda checked, p=param, s=-step: self.incrementParameter(p, s))
            
            btn_layout.addWidget(up_btn)
            btn_layout.addWidget(down_btn)
            
            # Add widgets to layout
            top_row.addWidget(param_label)
            top_row.addStretch(1)  # Add a spacer to push controls to the right
            top_row.addWidget(text_input)
            top_row.addLayout(btn_layout)
            
            # Slider
            slider = QSlider(Qt.Horizontal)
            slider.setMinimum(int(min_val * scale))
            slider.setMaximum(int(max_val * scale))
            slider.setValue(int(self.params[param] * scale))
            slider.valueChanged.connect(lambda v, p=param, s=scale: self.updateParameter(p, v/s))
            self.sliders[param] = slider
            
            param_layout.addLayout(top_row)
            param_layout.addWidget(slider)
            layout.addWidget(param_widget)
        
        # Coordinate editing section
        coord_group = QGroupBox("Coordinates")
        coord_layout = QGridLayout(coord_group)

        self.coord_inputs = {
            'x': QLineEdit(), 'y': QLineEdit(),
            'ra': QLineEdit(), 'dec': QLineEdit()
        }

        coord_layout.addWidget(QLabel("X:"), 0, 0)
        coord_layout.addWidget(self.coord_inputs['x'], 0, 1)
        coord_layout.addWidget(QLabel("Y:"), 1, 0)
        coord_layout.addWidget(self.coord_inputs['y'], 1, 1)
        coord_layout.addWidget(QLabel("RA:"), 2, 0)
        coord_layout.addWidget(self.coord_inputs['ra'], 2, 1)
        coord_layout.addWidget(QLabel("Dec:"), 3, 0)
        coord_layout.addWidget(self.coord_inputs['dec'], 3, 1)
        
        for name, field in self.coord_inputs.items():
            field.returnPressed.connect(lambda n=name: self._onCoordinateChanged(n))

        layout.addWidget(coord_group)
        
        # Zoom control
        zoom_layout = QHBoxLayout()
        zoom_label = QLabel("Zoom Level:")
        self.zoom_display = QLabel("1.00x")
        self.zoom_display.setMinimumWidth(50)
        self.zoom_reset_btn = QPushButton("Reset")
        self.zoom_reset_btn.setMinimumWidth(80)

        zoom_layout.addWidget(zoom_label)
        zoom_layout.addWidget(self.zoom_display)
        zoom_layout.addStretch()
        zoom_layout.addWidget(self.zoom_reset_btn)
        layout.addLayout(zoom_layout)

        # Bottom Button: Reset Scaling
        reset_btn = QPushButton("Reset Scaling")
        reset_btn.setMinimumWidth(120)
        reset_btn.clicked.connect(self.resetDefaults)
        layout.addWidget(reset_btn, 0, Qt.AlignHCenter)

        layout.addStretch()

        # Store button references
        self.load_files_btn = load_files_btn
        self.save_stamp_btn = save_stamp_btn
        self.save_full_btn = save_full_btn
    
    def _onCoordinateChanged(self, name):
        """Handle when a coordinate field is edited."""
        value = self.coord_inputs[name].text()
        self.centerRequest.emit({name: value})
    
    def updateCoordinateDisplay(self, x, y, ra_str, dec_str):
        """Update coordinate display fields from mouseover."""
        # Only update if the field doesn't have focus, to avoid interrupting user
        if not self.coord_inputs['x'].hasFocus():
            self.coord_inputs['x'].setText(str(x))
        if not self.coord_inputs['y'].hasFocus():
            self.coord_inputs['y'].setText(str(y))
        if not self.coord_inputs['ra'].hasFocus():
            self.coord_inputs['ra'].setText(ra_str.replace("RA: ", ""))
        if not self.coord_inputs['dec'].hasFocus():
            self.coord_inputs['dec'].setText(dec_str.replace("Dec: ", ""))

    def updateParameter(self, param, value):
        """Update parameter and emit change signal"""
        self.params[param] = value
        
        # Update display with proper formatting from config
        decimals = self.param_configs[param]['decimals']
        if decimals == 0:
            display_text = f"{int(value)}"
        else:
            display_text = f"{value:.{decimals}f}"
        
        if param in self.text_inputs:
            self.text_inputs[param].setText(display_text)
        
        self.parametersChanged.emit()
    
    def updateFromText(self, param):
        """Update parameter from text input"""
        if param in self.text_inputs:
            try:
                text_value = self.text_inputs[param].text()
                config = self.param_configs[param]
                decimals = config['decimals']
                
                if decimals == 0:
                    value = int(float(text_value))
                else:
                    value = float(text_value)
                
                # Apply bounds checking
                min_val = config['min']
                max_val = self.sliders[param].maximum() / config['scale'] if param in ['xc', 'yc'] else config['max']
                value = max(min_val, min(max_val, value))
                
                # Update parameter and slider
                self.params[param] = value
                self.sliders[param].setValue(int(value * config['scale']))
                
                # Update display
                self.updateParameter(param, value)
                
            except ValueError:
                # Restore previous value if invalid input
                config = self.param_configs[param]
                decimals = config['decimals']
                if decimals == 0:
                    self.text_inputs[param].setText(f"{int(self.params[param])}")
                else:
                    self.text_inputs[param].setText(f"{self.params[param]:.{decimals}f}")
    
    def incrementParameter(self, param, step):
        """Increment/decrement parameter by step"""
        current_value = self.params[param]
        new_value = current_value + step
        
        # Apply bounds checking from config
        config = self.param_configs[param]
        min_val = config['min']
        max_val = self.sliders[param].maximum() / config['scale'] if param in ['xc', 'yc'] else config['max']
        new_value = max(min_val, min(max_val, new_value))
        
        # Update parameter, slider, and display
        self.params[param] = new_value
        self.sliders[param].setValue(int(new_value * config['scale']))
        self.updateParameter(param, new_value)
    
    def setImageBounds(self, width, height):
        """Set slider ranges based on image dimensions"""
        # This no longer needs to set xc/yc sliders
        pass
    
    def setRegion(self, xlo, ylo, xhi, yhi):
        """Update parameters from region selection"""
        # This no longer needs to set xc/yc sliders
        self.params['sample_size'] = min(xhi - xlo, yhi - ylo)
        
        self.sliders['sample_size'].blockSignals(True)
        self.sliders['sample_size'].setValue(self.params['sample_size'])
        self.sliders['sample_size'].blockSignals(False)
        
        self.updateParameter('sample_size', self.params['sample_size'])
    
    def resetDefaults(self):
        """Reset all parameters to defaults"""
        defaults = {
            'noiselum': 0.15,
            'satpercent': 0.01,
            'noisesig': 1.0,
            'noisefloorsig': 2.0,
            'color_saturation': 2.0,
            'sample_size': 1000,
        }
        
        # Reset regular parameters
        for param, value in defaults.items():
            if param in self.sliders:
                self.params[param] = value
                slider = self.sliders[param]
                
                # Update slider value
                if param == 'sample_size':
                    slider.setValue(int(value))
                elif param == 'satpercent':
                    slider.setValue(int(value * 1000))
                else:
                    slider.setValue(int(value * 10))
                
                # Update display
                self.updateParameter(param, value)
        
        # Position is no longer reset from here; it's handled by recentering
        # to the middle of the image on load.
        print("✅ Scaling parameters reset to defaults")
    
    def saveImage(self):
        """Emit save signal"""
        # This will be connected to the main window's save function
        pass

    def updateZoomDisplay(self, zoom_level):
        """Update the zoom level display"""
        self.zoom_display.setText(f"{zoom_level:.2f}x")

    def getParameters(self):
        """Get current parameters from panel"""
        return self.params


class FinderView(QGraphicsView):
    """Interactive finder scope view"""
    panRequest = pyqtSignal(int, int) if QT_VERSION == "PyQt5" else Signal(int, int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.dragging = False
        self.full_image_size = None
        self.finder_image_size = None

    def set_image_sizes(self, full_size, finder_size):
        """Set image dimensions for coordinate conversion"""
        self.full_image_size = full_size
        self.finder_image_size = finder_size

    def mousePressEvent(self, event):
        """Handle mouse press to start panning"""
        if event.button() == Qt.LeftButton:
            self.dragging = True
            self.pan(event.pos())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        """Handle mouse move to pan"""
        if self.dragging:
            self.pan(event.pos())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        """Handle mouse release to stop panning"""
        if event.button() == Qt.LeftButton and self.dragging:
            self.dragging = False
        super().mouseReleaseEvent(event)

    def pan(self, pos):
        """Calculate and emit new center coordinates"""
        if not self.full_image_size or not self.finder_image_size:
            return

        # Convert view coordinates to scene coordinates
        scene_pos = self.mapToScene(pos)
        
        # Get finder pixmap dimensions (width, height)
        finder_nx, finder_ny = self.finder_image_size
        
        # Get full image dimensions (width, height)
        full_nx, full_ny = self.full_image_size

        if finder_nx == 0 or finder_ny == 0:
             return

        # Calculate ratios, clipping to avoid out-of-bounds clicks
        x_ratio = np.clip(scene_pos.x() / finder_nx, 0, 1)
        y_ratio = np.clip(scene_pos.y() / finder_ny, 0, 1)
        
        # Calculate new center in full image coordinates, inverting Y
        new_xc = int(full_nx * x_ratio)
        new_yc = int(full_ny * (1 - y_ratio))
        
        self.panRequest.emit(new_xc, new_yc)


# =============================================================================
# MAIN TRILOGY PYQT APPLICATION
# =============================================================================

class TrilogyPyQt(QMainWindow):
    """Main PyQt Trilogy application"""
    
    def __init__(self, image_pattern='../images/*_sci.fits', files_to_load=None):
        super().__init__()
        
        # Store image pattern for relaunch
        self.image_pattern = image_pattern
        
        self.image_files = []
        self.filters = []
        self.image_data = {}
        self.filter_colors = {}
        self.levels = {}
        self.scaled_images = {}
        
        # View center coordinates
        self.xc = 0
        self.yc = 0
        self.wcs_info = None
        self.reference_header = None

        # Finder scope attributes
        self.finder_view = None
        self.finder_scene = None
        self.finder_pixmap_item = None
        self.finder_rect_item = None
        self.low_res_image_data = {}
        self.low_res_scaled_images = {}
        self.finder_image = None
        self.finder_container = None
        
        # View state persistence
        self._initial_zoom_done = False
        
        # Initialize UI first (creates image_view)
        self.initUI()
        
        # Connect signals
        self.connectSignals()
        
        if files_to_load:
            # If files are provided on launch (e.g., test mode), load them
            self.loadImages(files_to_load)
            if self.filters:
                self.updateDisplay()
        else:
            # Otherwise, show file selection dialog on startup
            QTimer.singleShot(100, self.selectFiles)
    
    def initUI(self):
        """Initialize the user interface"""
        self.setWindowTitle("Trilogy PyQt - Fast Interactive Color Images from Astronomical FITS Data")
        self.setGeometry(100, 100, 1600, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout - THREE COLUMNS
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - Parameters
        self.param_panel = ParameterPanel()
        self.param_panel.setMaximumWidth(320)
        main_layout.addWidget(self.param_panel)
        
        # Center panel - Image
        image_widget = QWidget()
        image_layout = QVBoxLayout(image_widget)
        image_layout.setContentsMargins(0, 0, 0, 0)
        image_widget.setLayout(image_layout)
        
        # Image view
        self.image_view = ImageView()
        image_layout.addWidget(self.image_view)
        
        main_layout.addWidget(image_widget, 1)
        
        # Right panel - Filters
        self.filter_panel = FilterPanel()
        main_layout.addWidget(self.filter_panel)
        
        # Finder scope - positioned in resizeEvent, parented to central_widget
        self.finder_container = QFrame(central_widget)
        self.finder_container.setFrameShape(QFrame.StyledPanel)
        self.finder_container.setFixedSize(152, 152)
        self.finder_container.setStyleSheet("background-color: white; border: 1px solid black;")

        finder_layout = QVBoxLayout(self.finder_container)
        finder_layout.setContentsMargins(1, 1, 1, 1)

        self.finder_view = FinderView(self.finder_container)
        self.finder_scene = QGraphicsScene(self)
        self.finder_view.setScene(self.finder_scene)
        self.finder_view.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.finder_view.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.finder_view.setRenderHint(QPainter.Antialiasing, False)
        self.finder_view.setRenderHint(QPainter.SmoothPixmapTransform, True)
        finder_layout.addWidget(self.finder_view)
        
        # Initialize empty filter checkboxes dict for compatibility
        self.filter_checkboxes = {}
        
        # Status bar with coordinate display
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        
        # Create labels for coordinate display
        self.status_bar.clearMessage()
    
    def resizeEvent(self, event):
        """Handle window resize to reposition finder scope"""
        super().resizeEvent(event)
        if hasattr(self, 'finder_container') and self.finder_container:
            main_widget = self.centralWidget()
            if main_widget:
                margin = 10
                finder_size = self.finder_container.size()
                
                # Position finder scope in bottom-right of the main window
                x = main_widget.width() - finder_size.width() - margin
                y = main_widget.height() - finder_size.height() - margin
                self.finder_container.move(x, y)
                
                # Position button below finder scope
                if hasattr(self, 'update_finder_btn'):
                    btn_x = x
                    btn_y = y + finder_size.height() + 5 # 5px gap
                    self.update_finder_btn.setFixedSize(finder_size.width(), 30)
                    self.update_finder_btn.move(btn_x, btn_y)
    
    def setupFilterControls(self):
        """Setup filter controls after filters are loaded"""
        # Clear existing filters
        self.filter_panel.clearFilters()
        
        # Add filters to the new panel
        for filt in self.filters:
            color = self.filter_colors[filt]
            self.filter_panel.addFilter(filt, color)
        
        # Update compatibility dict for existing code
        self.filter_checkboxes = self.filter_panel.filter_checkboxes
        
        # Set image bounds for parameter panel
        if self.filters:
            first_filter = self.filters[0]
            ny, nx = self.image_data[first_filter].shape
            # self.param_panel.setImageBounds(nx, ny) # No longer needed
    
    def updateCoordinateDisplay(self, x, y, ra_str, dec_str):
        """Update coordinate display in status bar"""
        # This is now handled by the ParameterPanel
        pass
    
    def connectSignals(self):
        """Connect UI signals"""
        # Connect main parameter changes to a full refresh
        self.param_panel.parametersChanged.connect(self.refreshAllViews)
        self.param_panel.centerRequest.connect(self.onCenterRequest)
        
        # Connect view interactions
        self.image_view.positionChanged.connect(self.onPositionChanged)
        self.image_view.coordinatesChanged.connect(self.param_panel.updateCoordinateDisplay)
        self.image_view.zoomChanged.connect(self.updateFinderBox) # Only update finder box on zoom
        self.image_view.centerChanged.connect(self.onCenterChanged)
        
        # Connect filter panel signals to a full refresh
        self.filter_panel.filterChanged.connect(self.refreshAllViews)
        self.filter_panel.colorChanged.connect(self.onColorChanged)
        self.filter_panel.autoSetColors.connect(self.onAutoSetColors)
        
        # Connect save/load/reset buttons
        if hasattr(self.param_panel, 'save_stamp_btn') and self.param_panel.save_stamp_btn:
            self.param_panel.save_stamp_btn.clicked.connect(self.saveStamp)
        
        if hasattr(self.param_panel, 'save_full_btn') and self.param_panel.save_full_btn:
            self.param_panel.save_full_btn.clicked.connect(self.saveFullImage)
        
        if hasattr(self.param_panel, 'zoom_reset_btn') and self.param_panel.zoom_reset_btn:
            self.param_panel.zoom_reset_btn.clicked.connect(self.image_view.resetZoom)
        
        if hasattr(self.param_panel, 'load_files_btn') and self.param_panel.load_files_btn:
            self.param_panel.load_files_btn.clicked.connect(self.selectFiles)
        
        # Connect finder button
        if hasattr(self, 'update_finder_btn'):
            self.update_finder_btn.clicked.connect(self.updateFinderOverview)

        # Connect finder view pan signal
        if hasattr(self, 'finder_view'):
            self.finder_view.panRequest.connect(self.onFinderPan)
    
    def refreshAllViews(self):
        """Refresh both the main image display and the finder scope overview"""
        self.updateFinderOverview()
        self.updateDisplay()

    def updateDisplayWithViewPersistence(self, reset_scroll=False):
        """Update display - simplified without view state persistence"""
        self.updateDisplay()
    
    def resetZoomAndSaveState(self):
        """Reset zoom and update saved state"""
        self.image_view.resetZoom()
    
    def selectFiles(self):
        """Open file dialog to select FITS files"""
        try:
            # Open file dialog with multi-selection support
            files, _ = QFileDialog.getOpenFileNames(
                self,
                "Select FITS files (Shift+Click for multiple)",
                os.getcwd(),
                "FITS files (*.fits *.fit *.fts);;All files (*.*)"
            )
            
            if files:
                print(f"\n📁 Selected {len(files)} files:")
                for i, file in enumerate(files, 1):
                    print(f"   {i}. {os.path.basename(file)}")
                
                # Load the selected files
                self.loadImages(files)
                
                # Initial processing if filters were loaded
                if self.filters:
                    self.updateDisplay()
                    if not self._initial_zoom_done:
                        self.showEvent(QShowEvent()) # Trigger initial zoom logic
            else:
                print("❌ No files selected")
                
        except Exception as e:
            print(f"❌ Error selecting files: {e}")
            self.status_bar.showMessage(f"File selection error: {e}")
    
    def loadImages(self, file_list):
        """Load FITS images from file list"""
        if isinstance(file_list, str):
            # Handle old pattern-based loading for backward compatibility
            self.image_files = sorted(glob(file_list))
        else:
            # Handle new file list loading
            self.image_files = file_list
            
        if not self.image_files:
            print(f"❌ No valid image files provided")
            return
        
        # Extract filter names and sort by them to maintain wavelength order
        files_and_filters = extract_filter_names_from_files(self.image_files)
        
        # Sort by filter name
        files_and_filters.sort(key=lambda x: x[1])
        
        # Unpack sorted lists
        self.image_files = [item[0] for item in files_and_filters]
        self.filters = [item[1] for item in files_and_filters]
        
        # Remove 'total' if present
        valid_indices = [i for i, f in enumerate(self.filters) if f != 'total']
        self.image_files = [self.image_files[i] for i in valid_indices]
        self.filters = [self.filters[i] for i in valid_indices]
        
        print(f"\n🔍 Loading {len(self.filters)} filters:")
        
        # Load image data and try to get WCS info
        self.wcs_info = None
        pixel_scale = None
        
        for i, filt in enumerate(self.filters):
            try:
                hdu = fits.open(self.image_files[i])
                raw_data = hdu[0].data
                self.image_data[filt] = normalize_array_dtype(raw_data)
                
                # Get image dimensions
                ny, nx = raw_data.shape
                
                # Try to get WCS and pixel scale from first image
                if self.wcs_info is None:
                    try:
                        from astropy.wcs import WCS
                        self.wcs_info = WCS(hdu[0].header)
                        self.reference_header = hdu[0].header
                        
                        # Calculate pixel scale in arcsec/pixel
                        if hasattr(self.wcs_info, 'pixel_scale_matrix'):
                            pixel_scale_matrix = self.wcs_info.pixel_scale_matrix
                            pixel_scale = np.sqrt(np.abs(np.linalg.det(pixel_scale_matrix))) * 3600  # Convert to arcsec
                        elif 'CD1_1' in hdu[0].header and 'CD2_2' in hdu[0].header:
                            # Alternative method using CD matrix
                            cd1_1 = hdu[0].header['CD1_1']
                            cd2_2 = hdu[0].header['CD2_2']
                            pixel_scale = np.sqrt(abs(cd1_1 * cd2_2)) * 3600  # Convert to arcsec
                        elif 'CDELT1' in hdu[0].header and 'CDELT2' in hdu[0].header:
                            # Alternative method using CDELT
                            cdelt1 = hdu[0].header['CDELT1']
                            cdelt2 = hdu[0].header['CDELT2']
                            pixel_scale = np.sqrt(abs(cdelt1 * cdelt2)) * 3600  # Convert to arcsec
                        
                        if pixel_scale:
                            print(f"   ✅ WCS information loaded, pixel scale: {pixel_scale:.3f} arcsec/pixel")
                        else:
                            print(f"   ⚠️  WCS found but no pixel scale information")
                    except Exception as wcs_error:
                        print(f"   ⚠️  No WCS information found: {wcs_error}")
                
                # Report filter information
                size_arcsec_str = ""
                if pixel_scale:
                    width_arcsec = nx * pixel_scale
                    height_arcsec = ny * pixel_scale
                    size_arcsec_str = f" ({width_arcsec:.1f}\" × {height_arcsec:.1f}\")"
                
                print(f"   📊 {filt.upper()}: {nx} × {ny} pixels{size_arcsec_str}")
                
                hdu.close()
                
            except Exception as e:
                print(f"   ❌ Error loading {filt}: {e}")
                continue
        
        # Set WCS info for coordinate display
        if self.wcs_info:
            self.image_view.setWCSInfo(self.wcs_info)
        
        # Report summary
        if self.filters:
            print(f"\n📋 Summary:")
            print(f"   • Loaded {len(self.filters)} filters: {', '.join([f.upper() for f in self.filters])}")
            if pixel_scale:
                print(f"   • Pixel scale: {pixel_scale:.3f} arcsec/pixel")
            else:
                print(f"   • No pixel scale information available")
            
            # Center the view initially
            ny, nx = self.image_data[self.filters[0]].shape
            self.recenterView(nx // 2, ny // 2)

            # Assign rainbow colors and set up controls
            self.onAutoSetColors()
            
            # Setup filter controls now that filters are loaded
            self.setupFilterControls()

            # Generate finder image on initial load
            self.updateFinderImage()
            
            # Create the low-resolution image data once
            self.createLowResImages()
            
            # Defer the initial render of both views until the UI is ready
            QTimer.singleShot(100, self.refreshAllViews)
            
            # Trigger initial zoom if window is already visible
            if self.isVisible():
                self.showEvent(QShowEvent())
        else:
            print("❌ No valid filters loaded")
    
    def _assign_rainbow_colors_to_filters(self):
        """Assign rainbow colors to filters based on panel settings."""
        import matplotlib.cm as cm
        cmap = cm.get_cmap('rainbow')

        include_purple = self.filter_panel.purple_checkbox.isChecked()
        
        # Adjust range to exclude purple if checkbox is unchecked
        # rainbow colormap: 0.0=purple, ~0.15=blue, 1.0=red
        start_val = 0.05 if include_purple else 0.15
        end_val = 0.95
        color_range = end_val - start_val
        
        for i, filt in enumerate(self.filters):
            # Map filter index to the selected color range
            x = i / (len(self.filters) - 1) if len(self.filters) > 1 else 0
            mapped_x = start_val + x * color_range
            
            r, g, b, _ = cmap(mapped_x)
            self.filter_colors[filt] = np.array([r, g, b])

    def assignRainbowColors(self):
        """Assign rainbow colors to filters"""
        self._assign_rainbow_colors_to_filters()
    
    def onPositionChanged(self, dx, dy):
        """Handle position change from image view"""
        if self.filters and self.image_data:
            # Get image dimensions
            first_filter = self.filters[0]
            ny, nx = self.image_data[first_filter].shape
            
            # Update position with bounds checking
            new_xc = max(0, min(nx, self.xc + dx))
            new_yc = max(0, min(ny, self.yc + dy))
            
            self.recenterView(new_xc, new_yc)
            
            print(f"Position updated: xc={new_xc}, yc={new_yc}")
            
            # Update display, which also updates the finder box
            self.updateDisplay()
    
    def onZoomChanged(self, zoom_level):
        """Handle zoom change from image view"""
        # Note: zoom_level parameter is temporarily disabled for performance
        # self.param_panel.params['zoom_level'] = zoom_level
        # self.param_panel.updateParameter('zoom_level', zoom_level)
        
        # Update saved view state
        if self.image_view.image_item:
            self.saved_view_state = self.image_view.saveViewState()
        
        # Directly update the finder box on zoom change for responsiveness
        self.updateFinderBox()
        
        print(f"Zoom changed to: {zoom_level:.2f}x")
    
    def onColorChanged(self, filter_name, new_color):
        """Handle filter color change"""
        self.filter_colors[filter_name] = np.array(new_color)
        self.refreshAllViews()
        print(f"Filter {filter_name} color changed to {new_color}")
    
    def onAutoSetColors(self):
        """Automatically set colors for filters based on their names"""
        print("🎨 Auto-setting colors for filters...")
        self.filter_panel.clearFilters() # Clear existing filters to re-add them
        
        # Check color mode
        color_mode = self.filter_panel.color_mode
        n_filters = len(self.filters)
        include_purple = self.filter_panel.purple_checkbox.isChecked()

        # Special case: If in Rainbow mode with "Purple" unchecked and 3 filters, use RGB scheme
        if color_mode == "rainbow" and not include_purple and n_filters == 3:
            print("🌈 Rainbow mode with 3 filters and no purple, using BGR colors.")
            color_mode = "rgb" # Fall through to use the RGB logic
        
        if color_mode == "rgb":
            # RGB mode: first third blue, next third green, final third red
            n_filters = len(self.filters)
            third = n_filters // 3
            
            for i, filt in enumerate(self.filters):
                if i < third:
                    # First third: Blue
                    self.filter_colors[filt] = np.array([0.0, 0.0, 1.0])
                elif i < 2 * third:
                    # Second third: Green
                    self.filter_colors[filt] = np.array([0.0, 1.0, 0.0])
                else:
                    # Final third: Red
                    self.filter_colors[filt] = np.array([1.0, 0.0, 0.0])
                
                self.filter_panel.addFilter(filt, self.filter_colors[filt])
            
            print(f"✅ BGR auto-colors set: {third} blue, {third} green, {n_filters - 2*third} red")
        else:
            # Rainbow mode: Use the helper to assign colors based on the "Purple" checkbox
            self._assign_rainbow_colors_to_filters()
            for filt in self.filters:
                 self.filter_panel.addFilter(filt, self.filter_colors[filt])
            
            print("✅ Rainbow auto-colors set.")
        
        # Update compatibility dict for existing code
        self.filter_checkboxes = self.filter_panel.filter_checkboxes
        
        self.refreshAllViews()
    
    def getImageStamp(self, data, xc=None, yc=None, size=None):
        """Extract image stamp from data - ENSURES CONSISTENT SIZES"""
        ny, nx = data.shape
        
        if xc is None:
            xc = self.xc
        if yc is None:
            yc = self.yc
        if size is None:
            size = self.param_panel.params['sample_size']
            
        # Calculate bounds
        half_size = size // 2
        xlo = max(0, int(xc - half_size))
        xhi = min(nx, int(xc + half_size))
        ylo = max(0, int(yc - half_size))
        yhi = min(ny, int(yc + half_size))
        
        # **FIX: Ensure consistent stamp size by padding if needed**
        actual_width = xhi - xlo
        actual_height = yhi - ylo
        
        # Extract the available data
        stamp = data[ylo:yhi, xlo:xhi]
        
        # If stamp is smaller than requested size, pad with zeros
        if actual_width < size or actual_height < size:
            padded_stamp = np.zeros((size, size), dtype=data.dtype)
            
            # Calculate padding offsets
            pad_x = (size - actual_width) // 2
            pad_y = (size - actual_height) // 2
            
            # Place the actual data in the center of the padded array
            padded_stamp[pad_y:pad_y+actual_height, pad_x:pad_x+actual_width] = stamp
            stamp = padded_stamp
        
        # Ensure exact size by cropping if needed
        if stamp.shape[0] > size or stamp.shape[1] > size:
            stamp = stamp[:size, :size]
        
        return stamp, (xlo, xhi, ylo, yhi)
    
    def updateScaling(self):
        """Update scaling for all visible filters"""
        params = self.param_panel.params
        
        for filt in self.filters:
            if not self.filter_panel.isFilterChecked(filt):
                continue
                
            data = self.image_data[filt]
            stamp, extent = self.getImageStamp(data)
            
            unsatpercent = 100 - params['satpercent']
            
            x0, x1, x2 = fast_determine_scaling(
                stamp, unsatpercent, 
                params['noisesig'], 
                True,  # correctbias
                params['noisefloorsig']
            )
            
            self.levels[filt] = (x0, x1, x2)
            
            # Scale the stamp
            k = solve_k(x0, x1, x2, params['noiselum'])
            scaled = fast_imscale(stamp, x0, x1, x2, params['noiselum'], k)
            self.scaled_images[filt] = scaled
    
    def createColorImage(self):
        """Create RGB color image from scaled filter images"""
        if not self.scaled_images:
            return None
            
        try:
            # Get active filters using the new filter panel
            active_filters = [f for f in self.filters if self.filter_panel.isFilterChecked(f) and f in self.scaled_images]
            
            if not active_filters:
                return None
            
            # Calculate total luminosity for active filters
            rgb_sum = np.zeros(3)
            for filt in active_filters:
                # Get color from filter panel
                color = self.filter_panel.getFilterColor(filt)
                rgb_sum += np.array(color)
            
            if np.sum(rgb_sum) == 0:
                return None
            
            # Combine filters
            first_shape = next(iter(self.scaled_images.values())).shape
            rgb_total = np.zeros((3, *first_shape))
            
            for filt in active_filters:
                # Get color from filter panel
                color = np.array(self.filter_panel.getFilterColor(filt))
                color = color[:, np.newaxis, np.newaxis]
                rgb_total += color * self.scaled_images[filt]
            
            # Normalize
            rgb_average = rgb_total / rgb_sum[:, np.newaxis, np.newaxis]
            
            # Convert to uint8 and create PIL image
            imrgb = rgb_average.transpose(1, 2, 0).astype(np.uint8)
            im = Image.fromarray(imrgb, 'RGB')
            
            # Apply color saturation
            if self.param_panel.params['color_saturation'] > 1:
                im = ImageEnhance.Color(im).enhance(self.param_panel.params['color_saturation'])
                
            return np.array(im)
            
        except Exception as e:
            print(f"Error creating color image: {e}")
            return None
    
    def updateDisplay(self):
        """Update the image display"""
        try:
            self.updateScaling()
            color_image = self.createColorImage()
            
            if color_image is not None:
                # Calculate stamp offset for full image coordinates
                if self.filters:
                    first_filter = self.filters[0]
                    data = self.image_data[first_filter]
                    ny, nx = data.shape  # Full image dimensions
                    
                    # Get current stamp position
                    xc = self.xc
                    yc = self.yc
                    size = self.param_panel.params['sample_size']
                    
                    # Calculate stamp bounds (same logic as getImageStamp)
                    half_size = size // 2
                    xlo = max(0, int(xc - half_size))
                    ylo = max(0, int(yc - half_size))
                    
                    # Set stamp offset in image view
                    self.image_view.setStampOffset(xlo, ylo, nx, ny)
                
                # Keep the view centered on the new image content
                current_transform = self.image_view.transform()
                self.image_view.setImage(color_image)
                self.image_view.setTransform(current_transform, False) # `False` prevents signal emission

                # Update the finder box to reflect the new main view
                self.updateFinderBox()
            
        except Exception as e:
            print(f"Display update error: {e}")
            self.status_bar.showMessage(f"Error: {e}")
    
    def saveImage(self):
        """Legacy method - redirect to saveStamp"""
        self.saveStamp()
    
    def saveStamp(self):
        """Save the current stamp image"""
        try:
            original_color_image = self.createColorImage()
            if original_color_image is not None:
                # **FIX: Flip Y-axis for correct orientation in saved image**
                flipped_color_image = np.flip(original_color_image, axis=0)
                
                # Convert numpy array to PIL Image
                im = Image.fromarray(flipped_color_image)
                
                # Get save filename using a robust dialog
                dialog = QFileDialog(self, "Save Stamp Image", "trilogy_stamp.png")
                dialog.setAcceptMode(QFileDialog.AcceptSave)
                dialog.setNameFilter("PNG files (*.png);;JPEG files (*.jpg);;All files (*.*)")
                
                # On macOS, using the non-native dialog often fixes keyboard focus issues
                if sys.platform == "darwin":
                    dialog.setOption(QFileDialog.DontUseNativeDialog)

                if dialog.exec_() == QFileDialog.Accepted:
                    filename = dialog.selectedFiles()[0]
                    metadata = self._create_png_metadata(self.levels)
                    im.save(filename, pnginfo=metadata)
                    self.status_bar.showMessage(f"Stamp saved: {filename}")
                    print(f"✅ Stamp saved: {filename}")
                    
                    # Ask to save FITS
                    self._ask_and_save_rgb_fits(original_color_image, filename)
                else:
                    self.status_bar.showMessage("Save cancelled.")
            else:
                self.status_bar.showMessage("No image to save")
                
        except Exception as e:
            print(f"❌ Save stamp error: {e}")
            self.status_bar.showMessage(f"Save error: {e}")
    
    def saveFullImage(self):
        """Save the full image with current scaling parameters"""
        try:
            print("🔄 Generating full image with current parameters...")
            
            # Setup progress dialog
            active_filters = [f for f in self.filters if self.filter_panel.isFilterChecked(f)]
            if not active_filters:
                self.status_bar.showMessage("No active filters selected.")
                return

            progress_dialog = QProgressDialog(
                "Generating Full-Resolution Image...", 
                "Cancel", 0, len(active_filters), self
            )
            progress_dialog.setWindowModality(Qt.WindowModal)
            progress_dialog.setValue(0)
            progress_dialog.show()

            # Create full-size color image
            original_full_color_image, levels = self.createFullColorImage(progress_dialog)
            
            # Check if cancelled
            if progress_dialog.wasCanceled():
                print("❌ Full image generation cancelled.")
                self.status_bar.showMessage("Full image generation cancelled.")
                return

            if original_full_color_image is not None:
                # **FIX: Flip Y-axis for correct orientation in saved image**
                flipped_full_color_image = np.flip(original_full_color_image, axis=0)
                
                # Convert numpy array to PIL Image
                im = Image.fromarray(flipped_full_color_image)
                
                # Get save filename using a robust dialog
                dialog = QFileDialog(self, "Save Full Image", "trilogy_full_image.png")
                dialog.setAcceptMode(QFileDialog.AcceptSave)
                dialog.setNameFilter("PNG files (*.png);;JPEG files (*.jpg);;TIFF files (*.tiff);;All files (*.*)")
                
                # On macOS, using the non-native dialog often fixes keyboard focus issues
                if sys.platform == "darwin":
                    dialog.setOption(QFileDialog.DontUseNativeDialog)

                if dialog.exec_() == QFileDialog.Accepted:
                    filename = dialog.selectedFiles()[0]
                    print(f"💾 Saving full image to: {filename}")
                    metadata = self._create_png_metadata(levels)
                    im.save(filename, pnginfo=metadata)
                    self.status_bar.showMessage(f"Full image saved: {filename}")
                    print(f"✅ Full image saved: {filename}")
                    
                    # Ask to save FITS
                    self._ask_and_save_rgb_fits(original_full_color_image, filename)
                else:
                    self.status_bar.showMessage("Save cancelled.")
            else:
                self.status_bar.showMessage("No full image to save")
                print("❌ Failed to generate full image")
                
        except Exception as e:
            print(f"❌ Save full image error: {e}")
            import traceback
            traceback.print_exc()
            self.status_bar.showMessage(f"Save full image error: {e}")
    
    def createFullColorImage(self, progress_dialog=None):
        """Create RGB color image from full-size images using current parameters"""
        if not self.image_data:
            return None, None
            
        try:
            print("🎨 Processing full images with current scaling...")
            
            # Get active filters using the new filter panel
            active_filters = [f for f in self.filters if self.filter_panel.isFilterChecked(f)]
            
            if not active_filters:
                print("❌ No active filters")
                return None, None
            
            # Get full image dimensions
            first_filter = active_filters[0]
            ny, nx = self.image_data[first_filter].shape
            print(f"📐 Full image size: {nx} × {ny}")
            
            # Scale each active filter to full size
            scaled_full_images = {}
            params = self.param_panel.params
            levels_dict = {}
            
            for i, filt in enumerate(active_filters):
                if progress_dialog:
                    if progress_dialog.wasCanceled():
                        return None, None
                    progress_dialog.setLabelText(f"Processing {filt.upper()}...")
                    progress_dialog.setValue(i)
                    QApplication.processEvents() # Keep UI responsive

                print(f"   Processing {filt}...")
                
                # Get full image data
                full_data = self.image_data[filt]
                
                # Use current stamp for scaling determination
                stamp, _ = self.getImageStamp(full_data)
                
                # Calculate scaling parameters
                unsatpercent = 100 - params['satpercent']
                x0, x1, x2 = fast_determine_scaling(
                    stamp, unsatpercent, 
                    params['noisesig'], 
                    True,  # correctbias
                    params['noisefloorsig']
                )
                levels_dict[filt] = (x0, x1, x2)
                
                # Scale full image
                k = solve_k(x0, x1, x2, params['noiselum'])
                scaled_full = fast_imscale(full_data, x0, x1, x2, params['noiselum'], k)
                scaled_full_images[filt] = scaled_full
                print(f"   ✅ {filt} scaled to full size")
            
            if progress_dialog:
                progress_dialog.setLabelText("Combining images...")
                progress_dialog.setValue(len(active_filters))
                QApplication.processEvents()

            # Calculate total luminosity for active filters
            rgb_sum = np.zeros(3)
            for filt in active_filters:
                color = self.filter_panel.getFilterColor(filt)
                rgb_sum += np.array(color)
            
            if np.sum(rgb_sum) == 0:
                print("❌ No color information")
                return None, None
            
            # Combine filters for full image
            rgb_total = np.zeros((3, ny, nx))
            
            for filt in active_filters:
                color = np.array(self.filter_panel.getFilterColor(filt))
                color = color[:, np.newaxis, np.newaxis]
                rgb_total += color * scaled_full_images[filt]
            
            # Normalize
            rgb_average = rgb_total / rgb_sum[:, np.newaxis, np.newaxis]
            
            # Convert to uint8 and create PIL image
            imrgb = rgb_average.transpose(1, 2, 0).astype(np.uint8)
            
            # Apply color saturation
            if params['color_saturation'] > 1:
                im_pil = Image.fromarray(imrgb, 'RGB')
                im_pil = ImageEnhance.Color(im_pil).enhance(params['color_saturation'])
                imrgb = np.array(im_pil)
            
            print(f"✅ Full color image created: {imrgb.shape}")
            return imrgb, levels_dict
            
        except Exception as e:
            print(f"❌ Error creating full color image: {e}")
            import traceback
            traceback.print_exc()
            return None, None

    def showEvent(self, event):
        """Handle initial window show event for proper zoom"""
        super().showEvent(event)
        if not self._initial_zoom_done and self.image_view.image_item:
            # Defer the initial zoom until the window is fully sized
            QTimer.singleShot(0, self.resetZoomAndSaveState)
            self._initial_zoom_done = True

    def onCenterChanged(self, new_xc, new_yc):
        """Recenter the view based on a double-click event"""
        print(f"Recenter requested to: ({new_xc}, {new_yc})")
        
        # Consolidate recentering logic
        self.recenterView(new_xc, new_yc)

    def onFinderPan(self, new_xc, new_yc):
        """Handle panning request from the finder view"""
        print(f"Finder pan requested to: ({new_xc}, {new_yc})")
        
        # Consolidate recentering logic
        self.recenterView(new_xc, new_yc)

    def recenterView(self, new_xc, new_yc):
        """Unified method to recenter the view and update all UI elements"""
        if not (self.filters and self.image_data):
            return

        first_filter = self.filters[0]
        ny, nx = self.image_data[first_filter].shape
        
        # Update position with bounds checking
        self.xc = max(0, min(nx, int(new_xc)))
        self.yc = max(0, min(ny, int(new_yc)))
        
        # Update coordinate displays in the parameter panel
        ra_str, dec_str = self.image_view.pixelToRADec(self.xc, self.yc)
        self.param_panel.updateCoordinateDisplay(self.xc, self.yc, ra_str, dec_str)
        
        # Update the main display, which will also update the finder box
        self.updateDisplay()

    # =========================================================================
    # FINDER SCOPE METHODS
    # =========================================================================

    def createLowResImages(self, max_dim=150):
        """Create low-resolution versions of all filter images for the finder scope"""
        if not self.image_data:
            return

        print("Creating low-resolution images for finder scope...")
        self.low_res_image_data = {}
        
        for filt in self.filters:
            if filt in self.image_data:
                data = self.image_data[filt]
                ny, nx = data.shape
                
                if ny == 0 or nx == 0: continue

                # Determine downsampling factor
                if nx > ny:
                    factor = int(np.ceil(nx / max_dim))
                else:
                    factor = int(np.ceil(ny / max_dim))
                
                if factor <= 1:
                    self.low_res_image_data[filt] = data.copy()
                else:
                    # Simple non-interpolated subsampling
                    self.low_res_image_data[filt] = data[::factor, ::factor]

    def updateLowResScaling(self):
        """Scale low-resolution images based on current parameters"""
        if not self.low_res_image_data:
            return
            
        params = self.param_panel.getParameters()
        self.low_res_scaled_images = {}
        
        for filt, data in self.low_res_image_data.items():
            if self.filter_panel.isFilterChecked(filt):
                # Calculate levels from the entire low-res image for a better overview
                unsatpercent = 100 - params['satpercent']
                x0, x1, x2 = fast_determine_scaling(
                    data, unsatpercent, 
                    params['noisesig'], 
                    True,  # correctbias
                    params['noisefloorsig']
                )
                
                k = solve_k(x0, x1, x2, params['noiselum'])
                self.low_res_scaled_images[filt] = fast_imscale(data, x0, x1, x2, params['noiselum'], k)

    def createFinderColorImage(self):
        """Create RGB color image for the finder from scaled low-res filter images"""
        if not self.low_res_scaled_images:
            return None
                
        try:
            active_filters = [f for f in self.filters if self.filter_panel.isFilterChecked(f) and f in self.low_res_scaled_images]
            if not active_filters: return None
            
            rgb_sum = np.zeros(3)
            for filt in active_filters:
                color = self.filter_panel.getFilterColor(filt)
                rgb_sum += np.array(color)
            
            if np.sum(rgb_sum) == 0: return None
                
            first_shape = next(iter(self.low_res_scaled_images.values())).shape
            rgb_total = np.zeros((3, *first_shape))
                
            for filt in active_filters:
                color = np.array(self.filter_panel.getFilterColor(filt))
                color = color[:, np.newaxis, np.newaxis]
                rgb_total += color * self.low_res_scaled_images[filt]
                
            rgb_average = rgb_total / rgb_sum[:, np.newaxis, np.newaxis]
            imrgb = rgb_average.transpose(1, 2, 0).astype(np.uint8)
            im = Image.fromarray(imrgb, 'RGB')
            
            if self.param_panel.params['color_saturation'] > 1:
                im = ImageEnhance.Color(im).enhance(self.param_panel.params['color_saturation'])
                    
            return np.array(im)
                
        except Exception as e:
            print(f"Error creating finder color image: {e}")
            return None

    def updateFinderImage(self):
        """Generate and display the finder scope image"""
        print("Updating finder image...")
        self.createLowResImages()
        self.updateLowResScaling()
        self.finder_image = self.createFinderColorImage()

        if self.finder_image is not None:
            # Set image sizes for coordinate conversion in finder view
            if self.filters and self.image_data:
                full_ny, full_nx = self.image_data[self.filters[0]].shape
                finder_ny, finder_nx, _ = self.finder_image.shape
                self.finder_view.set_image_sizes((full_nx, full_ny), (finder_nx, finder_ny))

            self.finder_scene.clear()
            self.finder_pixmap_item = None
            self.finder_rect_item = None

            h, w, c = self.finder_image.shape
            qimage = QImage(self.finder_image.data, w, h, c * w, QImage.Format_RGB888)
            qimage = qimage.mirrored(False, True) # Flip to match main view
            pixmap = QPixmap.fromImage(qimage)

            self.finder_pixmap_item = self.finder_scene.addPixmap(pixmap)
            self.finder_scene.setSceneRect(self.finder_pixmap_item.boundingRect())
            self.finder_view.fitInView(self.finder_scene.sceneRect(), Qt.KeepAspectRatio)

            pen = QPen(Qt.red)
            pen.setWidth(2) # Make it thicker to be visible
            self.finder_rect_item = self.finder_scene.addRect(0, 0, 0, 0, pen)
            
            self.updateFinderBox()
        else:
            self.finder_scene.clear()
            self.finder_scene.setBackgroundBrush(QBrush(Qt.white))
            self.finder_pixmap_item = None
            self.finder_rect_item = None

    def updateFinderBox(self, *args):
        """Update the rectangle on the finder scope to show the current view"""
        if not all([self.finder_rect_item, self.finder_image is not None, self.filters]):
            return
        
        # Get full image dimensions
        first_filter = self.filters[0]
        if first_filter not in self.image_data: return
        full_ny, full_nx = self.image_data[first_filter].shape
        if full_nx == 0 or full_ny == 0: return

        # Get finder image dimensions and scaling
        finder_ny, finder_nx, _ = self.finder_image.shape
        x_scale = finder_nx / full_nx
        y_scale = finder_ny / full_ny

        # Get the visible part of the stamp from the main view's scene (top-down Y)
        visible_rect = self.image_view.mapToScene(self.image_view.viewport().rect()).boundingRect()
        
        # Get the stamp's offset (xlo, ylo) and size in FITS coordinates (bottom-up Y)
        xlo, ylo = self.image_view.stamp_offset
        stamp_size = self.param_panel.params['sample_size']

        # Convert the visible rect from stamp scene coordinates to full FITS coordinates
        # (vx, vy) is the top-left of the visible rect in the stamp scene
        vx, vy, vw, vh = visible_rect.x(), visible_rect.y(), visible_rect.width(), visible_rect.height()

        # FITS coordinates of the visible rectangle's bottom-left corner
        vis_x_fits = xlo + vx
        vis_y_fits = ylo + (stamp_size - (vy + vh))
        
        # Convert the FITS bounding box to the finder's scene coordinates (top-down Y)
        box_x = vis_x_fits * x_scale
        box_w = vw * x_scale
        box_h = vh * y_scale
        
        # The top of the box in the finder scene corresponds to the top of the rect in FITS
        vis_y_fits_top = vis_y_fits + vh
        box_y = (full_ny - vis_y_fits_top) * y_scale

        self.finder_rect_item.setRect(box_x, box_y, box_w, box_h)

    def updateFinderOverview(self):
        """Generate and display the finder scope image based on current settings"""
        print("Updating finder overview...")
        self.updateLowResScaling()
        self.finder_image = self.createFinderColorImage()

        if self.finder_image is not None:
            # Set image sizes for coordinate conversion in finder view
            if self.filters and self.image_data:
                full_ny, full_nx = self.image_data[self.filters[0]].shape
                finder_ny, finder_nx, _ = self.finder_image.shape
                self.finder_view.set_image_sizes((full_nx, full_ny), (finder_nx, finder_ny))

            self.finder_scene.clear()
            self.finder_pixmap_item = None
            self.finder_rect_item = None

            h, w, c = self.finder_image.shape
            qimage = QImage(self.finder_image.data, w, h, c * w, QImage.Format_RGB888)
            qimage = qimage.mirrored(False, True) # Flip to match main view
            pixmap = QPixmap.fromImage(qimage)

            self.finder_pixmap_item = self.finder_scene.addPixmap(pixmap)
            self.finder_scene.setSceneRect(self.finder_pixmap_item.boundingRect())
            self.finder_view.fitInView(self.finder_scene.sceneRect(), Qt.KeepAspectRatio)

            pen = QPen(Qt.red)
            pen.setWidth(2) # Make it thicker to be visible
            self.finder_rect_item = self.finder_scene.addRect(0, 0, 0, 0, pen)
            
            self.updateFinderBox()
        else:
            self.finder_scene.clear()
            self.finder_scene.setBackgroundBrush(QBrush(Qt.white))
            self.finder_pixmap_item = None
            self.finder_rect_item = None

    def onCenterRequest(self, coord_dict):
        """Handle a request to recenter from the parameter panel."""
        first_filter = self.filters[0]
        ny, nx = self.image_data[first_filter].shape
        
        new_xc, new_yc = self.xc, self.yc

        try:
            if 'x' in coord_dict:
                new_xc = int(float(coord_dict['x']))
            elif 'y' in coord_dict:
                new_yc = int(float(coord_dict['y']))
            elif 'ra' in coord_dict or 'dec' in coord_dict:
                if not self.wcs_info:
                    print("⚠️ Cannot recenter by RA/Dec: No WCS info available.")
                    return
                
                # Get current RA/Dec to fill in missing value
                current_ra_str, current_dec_str = self.image_view.pixelToRADec(self.xc, self.yc)
                
                ra_full_text = coord_dict.get('ra', current_ra_str.replace("RA: ", "").strip())
                dec_full_text = coord_dict.get('dec', current_dec_str.replace("Dec: ", "").strip())

                # Take only the part before '=', or the whole string if '=' is not present.
                ra_str_to_parse = ra_full_text.split('=')[0].strip()
                dec_str_to_parse = dec_full_text.split('=')[0].strip()
                
                # Use SkyCoord to parse flexible input
                if ':' in ra_str_to_parse:
                    coords = SkyCoord(f"{ra_str_to_parse} {dec_str_to_parse}", unit=(u.hourangle, u.deg))
                else:
                    coords = SkyCoord(f"{ra_str_to_parse} {dec_str_to_parse}", unit=(u.deg, u.deg))

                # Convert back to pixel coordinates
                new_xc, new_yc = self.wcs_info.world_to_pixel_values(coords.ra.deg, coords.dec.deg)
                new_xc, new_yc = int(new_xc), int(new_yc)

        except (ValueError, TypeError) as e:
            print(f"❌ Invalid coordinate input: {e}")
            # On error, just refresh the panel with current valid coordinates
            ra_str, dec_str = self.image_view.pixelToRADec(self.xc, self.yc)
            self.param_panel.updateCoordinateDisplay(self.xc, self.yc, ra_str, dec_str)
            return

        self.recenterView(new_xc, new_yc)

    def _create_png_metadata(self, levels):
        """Create PngInfo object with image generation parameters."""
        metadata = PngInfo()
        metadata.add_text("Created with Trilogy", "https://github.com/dancoe/trilogy")

        active_filters = [f for f in self.filters if self.filter_panel.isFilterChecked(f)]
        for filt in active_filters:
            color = self.filter_panel.getFilterColor(filt)
            try:
                rgb_hex = matplotlib.colors.to_hex(color)
                metadata.add_text(f'{filt}_rgb', rgb_hex)
            except Exception:
                metadata.add_text(f'{filt}_rgb_tuple', str(color))

        params = self.param_panel.getParameters()
        params['xc'] = self.xc
        params['yc'] = self.yc
        params['unsatpercent'] = 100 - params.get('satpercent', 0.01)

        params_to_save = [
            'sample_size', 'xc', 'yc', 'noiselum', 'satpercent', 'unsatpercent',
            'noisesig', 'noisefloorsig', 'color_saturation'
        ]

        for param in params_to_save:
            if param in params:
                metadata.add_text(param, str(params[param]))
        
        metadata.add_text('correctbias', 'True')

        for filt in active_filters:
            if filt in levels:
                for ix, level_val in enumerate(levels[filt]):
                    metadata.add_text(f'{filt}_x{ix}', str(level_val))
        
        return metadata

    def _ask_and_save_rgb_fits(self, rgb_image_array, source_png_filename):
        """Asks user if they want to save an RGB FITS file and saves it if so."""
        base, _ = os.path.splitext(source_png_filename)
        fits_filename_proposal = base + '_rgb.fits'

        reply = QMessageBox.question(self, 'Save RGB FITS',
                                     "Do you want to save an RGB FITS file of this image as well?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)

        if reply == QMessageBox.No:
            return

        dialog = QFileDialog(self, "Save RGB FITS Image", fits_filename_proposal)
        dialog.setAcceptMode(QFileDialog.AcceptSave)
        dialog.setNameFilter("FITS files (*.fits)")
        dialog.setDefaultSuffix("fits")
        if sys.platform == "darwin":
            dialog.setOption(QFileDialog.DontUseNativeDialog)

        if dialog.exec_() == QFileDialog.Accepted:
            fits_filename = dialog.selectedFiles()[0]
            if not fits_filename:
                return
            self._save_rgb_fits_file(rgb_image_array, fits_filename)

    def _save_rgb_fits_file(self, rgb_image_array, fits_filename):
        """Saves the given RGB numpy array as a FITS file."""
        try:
            print(f"💾 Saving RGB FITS file to: {fits_filename}")
            
            rgb_data_for_fits = np.transpose(rgb_image_array, (2, 0, 1))

            if self.reference_header:
                header = self.reference_header.copy()
            elif self.wcs_info:
                header = self.wcs_info.to_header()
            else:
                header = fits.Header()

            primary_hdu = fits.PrimaryHDU(header=header)
            hdul = fits.HDUList([primary_hdu])

            plane_names = ['RED', 'GREEN', 'BLUE']
            for i in range(3):
                image_hdu = fits.ImageHDU(data=rgb_data_for_fits[i], header=header)
                image_hdu.name = plane_names[i]
                hdul.append(image_hdu)
                
            hdul.writeto(fits_filename, overwrite=True)
            self.status_bar.showMessage(f"RGB FITS saved: {fits_filename}")
            print(f"✅ RGB FITS saved: {fits_filename}")

        except Exception as e:
            print(f"❌ Save RGB FITS error: {e}")
            import traceback
            traceback.print_exc()
            self.status_bar.showMessage(f"Save RGB FITS error: {e}")


# =============================================================================
# NOTEBOOK INTEGRATION FUNCTIONS
# =============================================================================

def launch_trilogy_pyqt(image_pattern='../images/*_sci.fits', files_to_load=None):
    """Launch PyQt Trilogy from Jupyter notebook"""
    
    # Create QApplication if it doesn't exist
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    
    # Create and show the main window
    window = TrilogyPyQt(image_pattern, files_to_load=files_to_load)
    window.show()
    
    # For notebook integration, don't start the event loop
    # The notebook's Qt event loop will handle it
    return window, app

def trilogy_pyqt_standalone(files_to_load=None):
    """Run PyQt Trilogy as standalone application"""
    app = QApplication(sys.argv)
    
    window = TrilogyPyQt(files_to_load=files_to_load)
    window.show()
    
    sys.exit(app.exec_())


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # --- Integrated launcher ---
    print("🚀 Launching Trilogy PyQt...")
    print("=" * 50)
    
    files_to_load = None
    if len(sys.argv) > 1 and sys.argv[1].lower() == 'test':
        print("🧪 Running in test mode, searching for FITS files...")
        search_paths = ['.', '..', os.path.join('..', 'images')]
        fits_files = []
        for path in search_paths:
            if os.path.isdir(path):
                # Search for common FITS extensions
                for ext in ['*.fits', '*.fit', '*.fts', '*_sci.fits']:
                    fits_files.extend(glob(os.path.join(path, ext)))
        
        if fits_files:
            unique_files = sorted(list(set(fits_files)))
            files_to_load = unique_files[:3]
            print(f"   ✅ Found {len(files_to_load)} files to load:")
            for f in files_to_load:
                print(f"      - {os.path.basename(f)}")
        else:
            print("   ❌ No FITS files found in standard locations (`.`, `..`, `../images`).")
            print("      Cannot run in test mode. Exiting.")
            sys.exit(1)
            
    try:
        print("\n✨ Features:")
        print("• Color Modes: RGB, Rainbow (with/without purple)")
        print("• Real-time parameter sliders and controls")
        print("• Smooth, hardware-accelerated pan and zoom")
        print("• Status bar with live (x,y) and RA/Dec coordinates")
        print("• Save current view (stamp) or full-resolution image")
        print("• Test mode: `python trilogy_pyqt.py test`")

        print("\n🎨 Usage Tips:")
        print("• Use 'Load Files' or run in test mode to begin")
        print("• Click filter color buttons to change colors")
        print("• Drag image to pan, use trackpad/wheel to zoom")
        
        print("\n✅ Application launching...")
        
        trilogy_pyqt_standalone(files_to_load=files_to_load)
        
    except Exception as e:
        print(f"❌ Launch failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1) 