"""
Visual detection system that identifies UI elements without relying on game metrics.
This focuses on pure computer vision approaches without knowledge of game semantics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple
from src.config.hardware_config import HardwareConfig
import cv2
import os
import logging
from skimage.metrics import structural_similarity

logger = logging.getLogger(__name__)

class VisualMetricsEstimator:
    def __init__(self, config: HardwareConfig):
        """Initialize visual detection system.
        
        Args:
            config (HardwareConfig): Hardware and training configuration
        """
        self.config = config
        self.device = config.get_device()
        self.dtype = config.get_dtype()
        
        # For menu detection
        self.menu_reference = None
        self.menu_matcher = cv2.SIFT_create()
        self.menu_flann = cv2.FlannBasedMatcher(dict(algorithm=1, trees=5), {})
        
        # Feature extraction for general use
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        ).to(self.device, dtype=self.dtype)
        
        # Global UI patterns (these are learned from experience without game knowledge)
        self.ui_patterns = {
            'menu': [],  # Stores feature vectors of known menu screens
            'normal_gameplay': []  # Stores feature vectors of normal gameplay
        }
        self.max_patterns = 50  # Maximum number of patterns to store
    
    def initialize_menu_detection(self, menu_reference_path):
        """Initialize menu detection with reference image."""
        if menu_reference_path and os.path.exists(menu_reference_path):
            try:
                self.menu_reference = cv2.imread(menu_reference_path, cv2.IMREAD_GRAYSCALE)
                if self.menu_reference is not None:
                    logger.info(f"Loaded menu reference image from {menu_reference_path}")
                    # Pre-compute keypoints and descriptors
                    self.menu_kp, self.menu_desc = self.menu_matcher.detectAndCompute(self.menu_reference, None)
                    return True
                else:
                    logger.warning(f"Failed to load menu reference image from {menu_reference_path}")
            except Exception as e:
                logger.error(f"Error loading menu reference: {e}")
        
        logger.warning("No valid menu reference image available")
        return False
    
    def setup_fallback_menu_detection(self):
        """Set up fallback menu detection based on color patterns and UI layout analysis.
        This method is used when no menu reference image is available.
        """
        logger.info("Setting up fallback menu detection system")
        
        # Common menu UI characteristics
        self.menu_color_ranges = [
            # Dark semi-transparent overlays (common in menus)
            ((0, 0, 0), (50, 50, 50)),
            # Blue UI elements (common in Cities Skylines UI)
            ((100, 50, 0), (255, 150, 50)),
            # White text
            ((200, 200, 200), (255, 255, 255))
        ]
        
        # Initialize UI pattern detection
        self.ui_pattern_threshold = 0.01  # Very aggressive threshold for menu detection
        self.menu_detection_initialized = True
        
        # Add specific patterns for Cities Skylines 2 main menu
        self.menu_templates = []
        
        # Add template for "RESUME GAME" button text - will be initialized later if needed
        self.resume_button_template = None
        self.has_direct_templates = False
        
        logger.info("Fallback menu detection initialized with enhanced sensitivity")
        
    def init_direct_template_matching(self):
        """Initialize direct template matching for Cities Skylines 2 menu elements."""
        try:
            # Try to load the resume button template
            btn_template_path = os.path.join(os.getcwd(), "resume_button_template.png")
            if os.path.exists(btn_template_path):
                self.resume_button_template = cv2.imread(btn_template_path, cv2.IMREAD_GRAYSCALE)
                if self.resume_button_template is not None:
                    logger.info(f"Loaded resume button template from {btn_template_path}")
                    self.has_direct_templates = True
            
            # Create a basic template for the white text if no template exists
            if self.resume_button_template is None:
                # Create a basic template that looks for white text on dark background
                template_height, template_width = 40, 150  # Approximate size of "RESUME GAME" text
                self.resume_button_template = np.zeros((template_height, template_width), dtype=np.uint8)
                # Add horizontal line in center to represent text
                self.resume_button_template[template_height//2-5:template_height//2+5, 10:-10] = 255
                logger.info("Created synthetic resume button template")
                self.has_direct_templates = True
            
            # Add cities skylines logo detection (white logo on dark background)
            logo_template_path = os.path.join(os.getcwd(), "cs2_logo_template.png")
            if os.path.exists(logo_template_path):
                logo_template = cv2.imread(logo_template_path, cv2.IMREAD_GRAYSCALE)
                if logo_template is not None:
                    self.menu_templates.append(logo_template)
                    logger.info(f"Loaded CS2 logo template from {logo_template_path}")
        except Exception as e:
            logger.error(f"Error initializing direct template matching: {e}")
    
    def detect_menu_fallback(self, frame: torch.Tensor) -> bool:
        """Fallback method for menu detection based on visual features.
        
        Args:
            frame (torch.Tensor): Input frame
            
        Returns:
            bool: True if menu detected, False otherwise
        """
        if frame is None:
            return False
            
        # Convert frame to numpy array for OpenCV processing
        if isinstance(frame, torch.Tensor):
            frame_np = frame.detach().cpu().numpy()
            if len(frame_np.shape) == 3:  # CHW format
                frame_np = frame_np.transpose(1, 2, 0)  # Convert to HWC
        else:
            frame_np = frame
            
        # Ensure frame is in proper format
        if frame_np is None or frame_np.size == 0 or frame_np.max() == 0:
            return False
            
        # First try color-based detection with the specific menu colors
        if len(frame_np.shape) == 3 and frame_np.shape[2] == 3:
            is_menu_by_color = self.detect_menu_by_colors(frame_np)
            if is_menu_by_color:
                logger.debug("Menu detected by color analysis")
                return True
            
        # Convert to grayscale and proper format
        if len(frame_np.shape) == 3 and frame_np.shape[2] == 3:
            gray = cv2.cvtColor(frame_np.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            gray = frame_np.astype(np.uint8)
            
        # Detect edges for UI elements with lower thresholds for higher sensitivity
        edges = cv2.Canny(gray, 30, 120)  # Lower thresholds to detect more edges
        
        # Calculate UI element coverage percentage
        ui_percentage = np.sum(edges > 0) / edges.size
        self.ui_pattern_threshold = 0.01  # Further lowered threshold for more sensitivity
        
        # Look for horizontal lines (common in UI) with reduced threshold
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=80, maxLineGap=15)
        
        horizontal_lines = 0
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if angle < 15 or angle > 165:  # More lenient angle for horizontal lines
                    horizontal_lines += 1
        
        # Check for solid color areas (common in menus) using multiple thresholds
        large_rectangles = 0
        for threshold in [150, 200]:  # Try multiple thresholds
            _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                area = w * h
                if area > (gray.shape[0] * gray.shape[1] * 0.03):  # Reduced threshold to 3% of screen
                    large_rectangles += 1
                    
        # Check for medium rectangles too (buttons, dialog boxes)
        medium_rectangles = 0
        _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h
            if area > (gray.shape[0] * gray.shape[1] * 0.01) and area < (gray.shape[0] * gray.shape[1] * 0.05):
                # Areas between 1% and 5% of screen (typical for buttons)
                medium_rectangles += 1
                
        # Add text detection with higher sensitivity
        text_proxy = 0
        kernel = np.ones((5,5), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        if lines is not None:
            vertical_line_mask = np.zeros_like(edges)
            for line in lines:
                x1, y1, x2, y2 = line[0]
                angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                if 75 < angle < 105:  # More lenient angle for vertical lines
                    cv2.line(vertical_line_mask, (x1, y1), (x2, y2), 255, 2)
            
            # Count potential text areas
            text_proxy = np.sum(vertical_line_mask & dilated_edges) / np.sum(dilated_edges + 1e-8)
        
        # Add center square check - many menus have content in the center of screen
        height, width = gray.shape
        center_region = gray[height//4:3*height//4, width//4:3*width//4]
        center_edges = cv2.Canny(center_region, 30, 120)
        center_edge_density = np.sum(center_edges > 0) / center_edges.size
        
        # Decision based on combined factors - much more aggressive detection
        is_menu = (ui_percentage > self.ui_pattern_threshold) or \
                  (horizontal_lines >= 2) or \
                  (large_rectangles >= 1) or \
                  (medium_rectangles >= 3) or \
                  (text_proxy > 0.05) or \
                  (center_edge_density > 0.03)
        
        # Log detection results
        logger.debug(f"Menu detection stats: ui_pct={ui_percentage:.4f}, h_lines={horizontal_lines}, " 
                    f"lg_rect={large_rectangles}, med_rect={medium_rectangles}, "
                    f"text={text_proxy:.4f}, center_density={center_edge_density:.4f}, is_menu={is_menu}")
        
        return is_menu
        
    def detect_main_menu(self, frame: torch.Tensor) -> bool:
        """Detect if current frame shows the main menu.
        
        Args:
            frame (torch.Tensor): Current frame observation
            
        Returns:
            bool: True if main menu is detected, False otherwise
        """
        # First try direct template matching for most reliable detection
        if hasattr(self, 'has_direct_templates') and self.has_direct_templates:
            # Try direct template matching first as it's most reliable
            is_menu = self.direct_template_match(frame)
            if is_menu:
                logger.debug("Menu detected via direct template matching")
                return True
            
        # Then try standard detection if reference image is available
        if self.menu_reference is not None:
            try:
                # Use existing feature matching method
                # Check if frame is None or empty
                if frame is None:
                    return False
                    
                # Convert to numpy for OpenCV processing
                if isinstance(frame, torch.Tensor):
                    frame_np = frame.detach().cpu().numpy()
                    if len(frame_np.shape) == 3:  # CHW format
                        frame_np = frame_np.transpose(1, 2, 0)  # Convert to HWC
                else:
                    frame_np = frame
                    
                # Check if frame is empty
                if frame_np is None or frame_np.size == 0 or np.all(frame_np == 0):
                    return False
                
                # Try color-based detection first as it's faster and potentially more reliable
                if len(frame_np.shape) == 3 and frame_np.shape[2] == 3:
                    is_menu = self.detect_menu_by_colors(frame_np)
                    if is_menu:
                        logger.debug("Menu detected via color analysis")
                        return True
                    
                # Convert to grayscale for feature matching
                if len(frame_np.shape) == 3 and frame_np.shape[2] == 3:
                    frame_gray = cv2.cvtColor(frame_np, cv2.COLOR_RGB2GRAY)
                else:
                    # If already grayscale, ensure it's the correct type
                    if len(frame_np.shape) == 2:
                        frame_gray = frame_np
                    else:
                        # Unexpected format, return false
                        return False
                        
                # Ensure frame is 8-bit unsigned integer (CV_8U)
                if frame_gray.dtype != np.uint8:
                    frame_gray = frame_gray.astype(np.uint8)
                    
                # If we have a reference image, use feature matching
                if self.menu_reference is not None and hasattr(self, 'menu_kp') and hasattr(self, 'menu_desc'):
                    try:
                        # Extract features from current frame
                        kp, desc = self.menu_matcher.detectAndCompute(frame_gray, None)
                        
                        # Not enough features for matching
                        if desc is None or len(kp) < 5:  # Lower minimum keypoints
                            # If not enough features, still try template matching
                            if np.array_equal(frame_gray.shape, self.menu_reference.shape):
                                # Direct template matching if shapes match
                                similarity = cv2.matchTemplate(
                                    frame_gray, self.menu_reference, cv2.TM_CCOEFF_NORMED
                                ).max()
                                if similarity > 0.5:  # Threshold for similarity
                                    return True
                            return False
                            
                        # Match features
                        matches = self.menu_flann.knnMatch(self.menu_desc, desc, k=2)
                        
                        # Apply ratio test
                        good_matches = []
                        for m, n in matches:
                            if m.distance < 0.8 * n.distance:  # More lenient ratio
                                good_matches.append(m)
                                
                        # If enough matches, consider it a menu
                        match_threshold = 1  # Ultra sensitive - any match triggers detection
                        is_menu = len(good_matches) >= match_threshold
                        if is_menu:
                            logger.debug(f"Menu detected with {len(good_matches)} feature matches")
                        return is_menu
                    except cv2.error:
                        # If OpenCV processing fails, try fallback
                        return self.detect_menu_fallback(frame)
            except Exception as e:
                logger.warning(f"Standard menu detection failed: {e}, falling back to pattern-based detection")
                return self.detect_menu_fallback(frame)
        else:
            # Use fallback method
            return self.detect_menu_fallback(frame)
    
    def direct_template_match(self, frame: torch.Tensor) -> bool:
        """Use direct template matching for menu detection - most reliable method.
        
        Args:
            frame (torch.Tensor): Input frame
            
        Returns:
            bool: True if menu detected, False otherwise
        """
        try:
            # Convert frame to numpy array for OpenCV
            if isinstance(frame, torch.Tensor):
                frame_np = frame.detach().cpu().numpy()
                if len(frame_np.shape) == 3:  # CHW format
                    frame_np = frame_np.transpose(1, 2, 0)  # Convert to HWC
            else:
                frame_np = frame
                
            # Ensure proper format
            if frame_np is None or frame_np.size == 0:
                return False
                
            # Convert to grayscale
            if len(frame_np.shape) == 3 and frame_np.shape[2] == 3:
                frame_gray = cv2.cvtColor(frame_np.astype(np.uint8), cv2.COLOR_RGB2GRAY)
            else:
                frame_gray = frame_np.astype(np.uint8)
            
            # Try matching against "RESUME GAME" button template
            if self.resume_button_template is not None:
                # Apply template matching
                result = cv2.matchTemplate(frame_gray, self.resume_button_template, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(result)
                
                # If good match found, consider it a menu
                if max_val > 0.5:  # Threshold for matching
                    logger.debug(f"Resume button detected (confidence: {max_val:.2f})")
                    return True
            
            # Check if we have any other templates to try
            if hasattr(self, 'menu_templates') and self.menu_templates:
                for template in self.menu_templates:
                    result = cv2.matchTemplate(frame_gray, template, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(result)
                    if max_val > 0.5:
                        logger.debug(f"Menu element detected (confidence: {max_val:.2f})")
                        return True
            
            # Also look for the specific menu layout characteristics
            # 1. Check for white text in center-left area (menu buttons)
            height, width = frame_gray.shape
            menu_region = frame_gray[int(height*0.3):int(height*0.7), int(width*0.2):int(width*0.5)]
            
            if menu_region.size > 0:
                # Threshold to find white text
                _, menu_text = cv2.threshold(menu_region, 200, 255, cv2.THRESH_BINARY)
                white_pixel_ratio = np.sum(menu_text > 0) / menu_region.size
                
                # If enough white pixels in menu region, likely a menu
                if white_pixel_ratio > 0.05:
                    logger.debug(f"Menu text area detected (white ratio: {white_pixel_ratio:.2f})")
                    return True
                    
            # Last check: Look for Cities Skylines logo (white hexagon) in center top
            logo_region = frame_gray[int(height*0.1):int(height*0.3), int(width*0.4):int(width*0.6)]
            if logo_region.size > 0:
                # Threshold to isolate bright logo
                _, logo_binary = cv2.threshold(logo_region, 200, 255, cv2.THRESH_BINARY)
                logo_pixel_ratio = np.sum(logo_binary > 0) / logo_region.size
                
                if logo_pixel_ratio > 0.1:
                    logger.debug("Potential CS2 logo detected in center top")
                    return True
                    
            return False
        except Exception as e:
            logger.error(f"Error in direct template matching: {e}")
            return False
    
    def save_current_frame_as_menu_reference(self, frame: torch.Tensor, save_path: str) -> bool:
        """Save current frame as a menu reference.
        
        Args:
            frame (torch.Tensor): Current frame
            save_path (str): Path to save the reference image
            
        Returns:
            bool: Success flag
        """
        try:
            # Convert to numpy
            if isinstance(frame, torch.Tensor):
                frame_np = frame.detach().cpu().numpy()
                if len(frame_np.shape) == 3:  # CHW format
                    frame_np = frame_np.transpose(1, 2, 0)  # Convert to HWC
            else:
                frame_np = frame
                
            # Convert to grayscale if needed
            if len(frame_np.shape) == 3 and frame_np.shape[2] == 3:
                frame_gray = cv2.cvtColor(frame_np, cv2.COLOR_RGB2GRAY)
            else:
                frame_gray = frame_np
            
            # Save the grayscale image
            cv2.imwrite(save_path, frame_gray)
            
            # Update the reference
            self.menu_reference = frame_gray
            self.menu_kp, self.menu_desc = self.menu_matcher.detectAndCompute(self.menu_reference, None)
            
            logger.info(f"Saved menu reference to {save_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving menu reference: {e}")
            return False
            
    def update_model(self, frame: torch.Tensor, reward: float):
        """Update any internal models based on rewards (placeholder).
        
        Args:
            frame (torch.Tensor): Current frame
            reward (float): The reward received
        """
        # This is just a placeholder - we don't need implementation since
        # our autonomous system handles learning separately
        pass
        
    def calculate_reward(self, frame):
        """Calculate a simple placeholder reward (for compatibility).
        
        Args:
            frame: The current observation frame
            
        Returns:
            float: A small constant reward
        """
        # This is just a placeholder - we only keep it for backward compatibility
        # The actual rewards come from the autonomous reward system
        return 0.01 

    def detect_menu_by_colors(self, frame_np: np.ndarray) -> bool:
        """Detect menu by checking for specific menu colors.
        
        Args:
            frame_np (np.ndarray): Input frame as numpy array in HWC format
            
        Returns:
            bool: True if menu detected, False otherwise
        """
        try:
            # Ensure frame is in BGR format for OpenCV
            frame_bgr = frame_np.astype(np.uint8)
            if frame_bgr.shape[2] != 3:
                return False
                
            # Define exact menu colors from user-provided hex values
            # These are the colors found in the Cities Skylines 2 menu
            dark_green_rgb = (38, 42, 29)  # #262A1D - Dark green background
            gray_rgb = (124, 129, 119)     # #7C8177 - Medium gray with green tint
            
            # Convert RGB to BGR for OpenCV
            dark_green_bgr = (dark_green_rgb[2], dark_green_rgb[1], dark_green_rgb[0])
            gray_bgr = (gray_rgb[2], gray_rgb[1], gray_rgb[0])
            
            # Define color ranges with tolerance
            tolerance = 15
            
            # Create lower and upper bounds for dark green
            dark_green_lower = np.array([max(0, dark_green_bgr[i] - tolerance) for i in range(3)])
            dark_green_upper = np.array([min(255, dark_green_bgr[i] + tolerance) for i in range(3)])
            
            # Create lower and upper bounds for gray
            gray_lower = np.array([max(0, gray_bgr[i] - tolerance) for i in range(3)])
            gray_upper = np.array([min(255, gray_bgr[i] + tolerance) for i in range(3)])
            
            # Create color masks
            dark_green_mask = cv2.inRange(frame_bgr, dark_green_lower, dark_green_upper)
            gray_mask = cv2.inRange(frame_bgr, gray_lower, gray_upper)
            
            # Calculate percentage of pixels matching each color
            dark_green_percentage = np.sum(dark_green_mask > 0) / dark_green_mask.size
            gray_percentage = np.sum(gray_mask > 0) / gray_mask.size
            
            # Log the percentages
            logger.debug(f"Menu color detection: Dark green={dark_green_percentage:.4f}, Gray={gray_percentage:.4f}")
            
            # Check if the percentages exceed thresholds
            # Menu typically has a significant amount of these colors
            # Using a lower threshold since we're looking for very specific colors
            return (dark_green_percentage > 1.00) or (gray_percentage > 1.00)
            
        except Exception as e:
            logger.error(f"Error in color-based menu detection: {e}")
            return False

    def find_resume_game_button(self, frame_np: np.ndarray) -> Tuple[bool, Tuple[int, int]]:
        """Find the 'RESUME GAME' button in the frame using multiple detection methods.
        
        Args:
            frame_np (np.ndarray): Input frame as numpy array in HWC format
            
        Returns:
            Tuple[bool, Tuple[int, int]]: Success flag and button coordinates (x, y)
        """
        try:
            if frame_np is None or frame_np.size == 0:
                return False, (0, 0)
                
            # Ensure frame is in proper format for processing
            frame = frame_np.astype(np.uint8)
            height, width = frame.shape[:2]
            
            # 1. Method: Look for white text in the expected button region
            # Convert to grayscale for text detection
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
                
            # Define regions where "RESUME GAME" might appear (multiple candidates)
            candidate_regions = [
                # Center region - most common location
                (int(width*0.35), int(height*0.3), int(width*0.65), int(height*0.5)),
                # Lower center region - alternative location
                (int(width*0.35), int(height*0.5), int(width*0.65), int(height*0.7)),
                # Upper center region - another possibility
                (int(width*0.35), int(height*0.2), int(width*0.65), int(height*0.4))
            ]
            
            # Check each candidate region
            for x1, y1, x2, y2 in candidate_regions:
                region = gray[y1:y2, x1:x2]
                if region.size == 0:
                    continue
                    
                # Apply binary thresholding to isolate white text
                _, binary = cv2.threshold(region, 200, 255, cv2.THRESH_BINARY)
                
                # Filter components by size to find text-like regions
                nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
                
                # Find horizontally aligned components that might form text
                text_candidates = []
                for i in range(1, nlabels):  # Skip background (0)
                    x, y, w, h, area = stats[i]
                    if 10 < w < 150 and 10 < h < 50 and area > 100:
                        text_candidates.append((x, y, w, h, centroids[i]))
                
                # Horizontal grouping of components to identify text lines
                if len(text_candidates) >= 3:  # Need multiple components for a text
                    # Sort by y-coordinate to group by lines
                    text_candidates.sort(key=lambda c: c[1])
                    
                    # Group components that are roughly on the same line (y-coordinate)
                    y_threshold = 10  # Max vertical distance for same line
                    lines = []
                    current_line = [text_candidates[0]]
                    
                    for i in range(1, len(text_candidates)):
                        if abs(text_candidates[i][1] - current_line[0][1]) < y_threshold:
                            current_line.append(text_candidates[i])
                        else:
                            if len(current_line) >= 2:  # Line with at least 2 components
                                lines.append(current_line)
                            current_line = [text_candidates[i]]
                    
                    if len(current_line) >= 2:
                        lines.append(current_line)
                    
                    # Examine lines that might contain "RESUME GAME"
                    for line in lines:
                        # Sort by x-coordinate to get left-to-right order
                        line.sort(key=lambda c: c[0])
                        
                        # Calculate line width and length - "RESUME GAME" is relatively wide
                        line_width = line[-1][0] + line[-1][2] - line[0][0]
                        if line_width > 100:  # Minimum width for "RESUME GAME"
                            # Calculate center of this line
                            center_x = line[0][0] + line_width // 2
                            center_y = int(sum(c[1] for c in line) / len(line))
                            
                            # Adjust coordinates to original frame
                            button_x = x1 + center_x
                            button_y = y1 + center_y
                            
                            logger.info(f"Potential 'RESUME GAME' button detected at ({button_x}, {button_y})")
                            return True, (button_x, button_y)
            
            # 2. Method: Look for button-like shapes in the frame
            # Use edge detection to find button shapes
            edges = cv2.Canny(gray, 50, 150)
            dilated_edges = cv2.dilate(edges, np.ones((3,3), np.uint8), iterations=1)
            
            # Find contours that might be buttons
            contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours to find button-like shapes
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                
                # Check if the shape has reasonable button-like dimensions
                if 100 < w < 300 and 20 < h < 60:
                    # Calculate aspect ratio - buttons typically have specific aspect ratio
                    aspect_ratio = w / h
                    if 3 < aspect_ratio < 10:  # Buttons typically wider than tall
                        # Now check if this area contains white text
                        button_region = gray[y:y+h, x:x+w]
                        _, button_binary = cv2.threshold(button_region, 200, 255, cv2.THRESH_BINARY)
                        white_ratio = np.sum(button_binary > 0) / button_binary.size
                        
                        # Buttons with white text will have a certain percentage of white pixels
                        if 0.1 < white_ratio < 0.5:
                            button_x = x + w // 2
                            button_y = y + h // 2
                            logger.info(f"Button-like shape detected at ({button_x}, {button_y})")
                            return True, (button_x, button_y)
            
            # 3. Method: Use common known positions for the "RESUME GAME" button
            # These are common positions based on multiple screenshots/resolutions
            common_positions = [
                (int(width * 0.45), int(height * 0.387)),   # From screenshot
                (int(width * 0.5), int(height * 0.4)),      # Center position
                (int(width * 0.5), int(height * 0.35)),     # Slightly higher
                (int(width * 0.5), int(height * 0.45))      # Slightly lower
            ]
            
            # Check each common position for indicators of a button
            for x, y in common_positions:
                # Define a small region around the position
                region_size = 30
                x1 = max(0, x - region_size)
                y1 = max(0, y - region_size)
                x2 = min(width, x + region_size)
                y2 = min(height, y + region_size)
                
                region = gray[y1:y2, x1:x2]
                if region.size == 0:
                    continue
                
                # Apply binary thresholding
                _, binary = cv2.threshold(region, 180, 255, cv2.THRESH_BINARY)
                white_ratio = np.sum(binary > 0) / binary.size
                
                # If there's a reasonable amount of white (text) in this region
                if 0.1 < white_ratio < 0.5:
                    logger.info(f"Potential button at common position ({x}, {y})")
                    return True, (x, y)
            
            # If all methods fail, return the most likely position (center of upper half)
            logger.warning("Could not detect 'RESUME GAME' button, using fallback position")
            return False, (width // 2, height // 3)
            
        except Exception as e:
            logger.error(f"Error finding 'RESUME GAME' button: {e}")
            return False, (0, 0) 