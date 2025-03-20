"""
Visual detection system that identifies UI elements without relying on game metrics.
This focuses on pure computer vision approaches without knowledge of game semantics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, List
from src.config.hardware_config import HardwareConfig
import cv2
import os
import logging
from src.environment.visual_change_analyzer import VisualChangeAnalyzer

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
        self.menu_detection_initialized = False
        
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
        
        # Initialize visual change analyzer
        self.visual_change_analyzer = VisualChangeAnalyzer()
    
    def initialize_menu_detection(self, menu_reference_path):
        """Initialize menu detection with reference image."""
        if menu_reference_path and os.path.exists(menu_reference_path):
            try:
                # Load reference image
                self.menu_reference = cv2.imread(menu_reference_path, cv2.IMREAD_GRAYSCALE)
                
                if self.menu_reference is not None:
                    logger.info(f"Loaded menu reference image from {menu_reference_path}")
                    
                    # Check if image is valid
                    if self.menu_reference.size == 0 or self.menu_reference.max() == 0:
                        logger.error("Menu reference image is blank or invalid")
                        return self.setup_fallback_menu_detection()
                    
                    # Pre-compute keypoints and descriptors
                    self.menu_kp, self.menu_desc = self.menu_matcher.detectAndCompute(self.menu_reference, None)
                    
                    if self.menu_kp is None or len(self.menu_kp) < 10 or self.menu_desc is None:
                        logger.warning("Few or no keypoints detected in menu reference image")
                        logger.info("Attempting to enhance reference image...")
                        
                        # Try to enhance the image with preprocessing
                        enhanced_ref = self.menu_reference.copy()
                        enhanced_ref = cv2.GaussianBlur(enhanced_ref, (3, 3), 0)
                        enhanced_ref = cv2.equalizeHist(enhanced_ref)
                        
                        # Try detecting features on enhanced image
                        self.menu_kp, self.menu_desc = self.menu_matcher.detectAndCompute(enhanced_ref, None)
                        
                        if self.menu_kp is None or len(self.menu_kp) < 10:
                            logger.warning("Failed to detect features even after enhancement. Using fallback detection.")
                            return self.setup_fallback_menu_detection()
                    
                    self.menu_detection_initialized = True
                    logger.info(f"Menu detection initialized with {len(self.menu_kp)} keypoints")
                    return True
                else:
                    logger.warning(f"Failed to load menu reference image from {menu_reference_path}")
            except Exception as e:
                logger.error(f"Error loading menu reference: {e}")
        
        logger.warning("No valid menu reference image available, using fallback detection")
        return self.setup_fallback_menu_detection()
    
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
            
        # Ensure frame is uint8 for OpenCV operations
        if frame_np.dtype != np.uint8:
            if frame_np.max() <= 1.0:
                frame_np = (frame_np * 255).astype(np.uint8)
            else:
                frame_np = frame_np.astype(np.uint8)
        
        # Check for version string in bottom left (reliable menu indicator)
        version_string_detected = self.detect_version_string(frame_np)
        if version_string_detected:
            logger.debug("Menu detected by version string in bottom left")
            return True
            
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
            
        # Apply blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Try multiple edge detection approaches
        edges1 = cv2.Canny(blurred, 30, 120)  # More sensitive
        edges2 = cv2.Canny(blurred, 50, 150)  # More selective
        
        # Combine edges for more robust detection
        edges = cv2.bitwise_or(edges1, edges2)
        
        # Dilate edges to connect nearby lines
        kernel = np.ones((3, 3), np.uint8)
        dilated_edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Calculate UI element coverage percentage
        ui_percentage = np.sum(edges > 0) / edges.size
        
        # Look for horizontal and vertical lines using both the original and dilated edges
        horizontal_lines_count = 0
        vertical_lines_count = 0
        large_rectangles = 0
        medium_rectangles = 0
        
        # Look for lines in the dilated edges
        try:
            lines = cv2.HoughLinesP(dilated_edges, 1, np.pi/180, threshold=50, minLineLength=40, maxLineGap=20)
            
            if lines is not None:
                # Count horizontal and vertical lines
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    
                    # Calculate angle
                    if abs(x2 - x1) < 1:  # Avoid division by zero
                        angle = 90.0
                    else:
                        angle = abs(np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi)
                        
                    # Classify lines with more lenient angles
                    if angle < 20:  # Horizontal-ish
                        horizontal_lines_count += 1
                    elif angle > 70:  # Vertical-ish
                        vertical_lines_count += 1
        except Exception as e:
            logger.warning(f"Error detecting lines: {e}")
        
        # Find contours to detect UI rectangles
        try:
            contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            height, width = gray.shape
            
            for contour in contours:
                # Get bounding rect
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / max(h, 1)
                
                # Rectangle size relative to frame
                rel_size = (w * h) / (width * height)
                
                # Classify rectangles
                if rel_size > 0.1 and 0.5 < aspect_ratio < 2.0:  # Large rectangles common in menus
                    large_rectangles += 1
                elif 0.02 < rel_size < 0.1 and 0.5 < aspect_ratio < 2.0:  # Medium sized UI elements
                    medium_rectangles += 1
        except Exception as e:
            logger.warning(f"Error detecting rectangles: {e}")
        
        # Text detection proxy (vertical lines intersecting with horizontal)
        text_proxy = 0.0
        if lines is not None:
            try:
                vertical_line_mask = np.zeros_like(edges)
                horizontal_line_mask = np.zeros_like(edges)
                
                for line in lines:
                    x1, y1, x2, y2 = line[0]
                    angle = np.abs(np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi)
                    if 75 < angle < 105:  # Vertical lines
                        cv2.line(vertical_line_mask, (x1, y1), (x2, y2), 255, 2)
                    elif angle < 15:  # Horizontal lines
                        cv2.line(horizontal_line_mask, (x1, y1), (x2, y2), 255, 2)
                
                # Combine vertical and horizontal lines - intersections hint at structured UI
                intersection_mask = cv2.bitwise_and(vertical_line_mask, horizontal_line_mask)
                text_proxy = np.sum(intersection_mask > 0) / max(1, np.sum(dilated_edges > 0))
            except Exception as e:
                logger.warning(f"Error calculating text proxy: {e}")
        
        # Check center region - menus often have content centered
        height, width = gray.shape
        center_region = gray[height//4:3*height//4, width//4:3*width//4]
        center_edges = cv2.Canny(center_region, 30, 120)
        center_edge_density = np.sum(center_edges > 0) / center_edges.size
        
        # Decision based on multiple factors - using weighted approach
        factors = [
            ui_percentage > 0.01,  # Very low threshold for edge density
            horizontal_lines_count >= 2,  # At least some horizontal lines
            vertical_lines_count >= 2,    # At least some vertical lines
            large_rectangles >= 1,        # Large UI panels
            medium_rectangles >= 2,       # UI elements
            text_proxy > 0.02,            # Text-like patterns
            center_edge_density > 0.02    # Content in center of screen
        ]
        
        # Count how many factors indicate a menu
        factor_count = sum(1 for factor in factors if factor)
        
        # Decision threshold - if at least 3 factors indicate a menu
        is_menu = factor_count >= 3
        
        # Advanced logging for diagnostics
        logger.debug(
            f"Menu detection stats: ui_pct={ui_percentage:.4f}, h_lines={horizontal_lines_count}, "
            f"v_lines={vertical_lines_count}, lg_rect={large_rectangles}, med_rect={medium_rectangles}, "
            f"text={text_proxy:.4f}, center_density={center_edge_density:.4f}, factors={factor_count}, is_menu={is_menu}"
        )
        
        return is_menu
        
    def detect_main_menu(self, frame: np.ndarray) -> bool:
        """Detect if the frame shows the main menu.
        
        Args:
            frame: Input frame
            
        Returns:
            bool: True if main menu detected
        """
        try:
            # Ensure frame is in proper format
            if frame is None or frame.size == 0:
                return False
            
            # Convert to numpy if it's a tensor
            if hasattr(frame, 'cpu') and hasattr(frame, 'numpy'):
                frame = frame.detach().cpu().numpy()
            
            # Convert float32 (0-1) to uint8 (0-255) if needed
            if frame.dtype == np.float32:
                frame = (frame * 255).astype(np.uint8)
            
            # Ensure frame has proper shape
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                if len(frame.shape) == 3:
                    # Handle channels first format
                    if frame.shape[0] == 3:
                        frame = frame.transpose(1, 2, 0)
                    else:
                        logger.error(f"Error in main menu detection: Invalid shape {frame.shape}")
                        return False
                else:
                    logger.error(f"Error in main menu detection: Invalid dimension {len(frame.shape)}")
                    return False
            
            # Use the menu detector if initialized
            if self.menu_detection_initialized:
                if hasattr(self, 'menu_flann') and self.menu_desc is not None:
                    # Convert to grayscale for feature matching
                    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    return self._match_menu_features(gray)
                else:
                    # Use fallback detection methods
                    return self.detect_menu_fallback(frame)
            else:
                # Initialize menu detection on first use
                self.setup_fallback_menu_detection()
                return self.detect_menu_fallback(frame)
        except Exception as e:
            logger.error(f"Error in main menu detection: {e}")
            return False
        
    def _match_menu_features(self, frame_np: np.ndarray) -> bool:
        """Detect menu using feature matching with reference screenshot.
        
        Args:
            frame_np (np.ndarray): Current frame as numpy array
            
        Returns:
            bool: True if menu detected, False otherwise
        """
        # Convert to grayscale for template matching
        if len(frame_np.shape) == 3:
            gray_frame = cv2.cvtColor(frame_np, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = frame_np
            
        try:
            # SIFT feature matching
            kp_frame, desc_frame = self.menu_matcher.detectAndCompute(gray_frame, None)
            
            # If no keypoints found, unlikely to be a menu
            if kp_frame is None or desc_frame is None or len(kp_frame) < 10:
                return False
                
            # Ensure menu descriptors exist
            if self.menu_desc is None or len(self.menu_desc) == 0:
                logger.warning("Menu reference descriptors not available")
                return False
                
            # Match against reference menu image
            matches = self.menu_flann.knnMatch(self.menu_desc, desc_frame, k=2)
            
            # Apply ratio test for good matches
            good_matches = []
            for match_pair in matches:
                if len(match_pair) != 2:
                    continue
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
                    
            # Calculate match ratio
            match_ratio = len(good_matches) / max(1, len(self.menu_kp))
            
            # Decision threshold - reduced to 10% for better detection sensitivity
            return match_ratio > 0.10
        except Exception as e:
            logger.warning(f"Error in SIFT template matching: {e}")
            
            # Fallback to simpler template matching if SIFT fails
            try:
                # Use simple template matching (less accurate but more robust)
                result = cv2.matchTemplate(gray_frame, self.menu_reference, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, _ = cv2.minMaxLoc(result)
                return max_val > 0.4  # Reduced threshold for better sensitivity
            except Exception as e2:
                logger.error(f"Template matching fallback also failed: {e2}")
                return False
                
    def detect_menu_by_colors(self, frame_np: np.ndarray) -> bool:
        """Detect potential menu screens by analyzing general UI patterns.
        
        Instead of using specific color values from CS2, this method looks for general
        UI patterns that might indicate a menu:
        - Large areas of uniform color (menu backgrounds)
        - High contrast text regions
        - Regular grid/alignment patterns
        
        Args:
            frame_np (np.ndarray): Input frame as numpy array in HWC format
            
        Returns:
            bool: True if menu-like UI detected, False otherwise
        """
        try:
            # Ensure frame is in BGR format for OpenCV
            frame_bgr = frame_np.astype(np.uint8)
            if frame_bgr.shape[2] != 3:
                return False
                
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            
            # 1. Check for large areas of uniform color (common in menus)
            # Apply slight blur to remove noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Calculate standard deviation in regions - low std dev = uniform areas
            regions_y, regions_x = 4, 4
            height, width = gray.shape
            region_h, region_w = height // regions_y, width // regions_x
            
            uniform_regions = 0
            for y in range(regions_y):
                for x in range(regions_x):
                    region = blurred[y*region_h:(y+1)*region_h, x*region_w:(x+1)*region_w]
                    std_dev = np.std(region)
                    if std_dev < 20:  # Low standard deviation = uniform color
                        uniform_regions += 1
            
            # 2. Look for text-like high contrast regions
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / (height * width)
            
            # 3. Check for aligned elements (common in menus)
            # Detect horizontal and vertical lines
            horizontal_lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, minLineLength=width//5, maxLineGap=20)
            vertical_lines = cv2.HoughLinesP(edges, 1, np.pi/2, 50, minLineLength=height//5, maxLineGap=20)
            
            has_aligned_elements = (horizontal_lines is not None and len(horizontal_lines) > 2) or \
                                  (vertical_lines is not None and len(vertical_lines) > 2)
            
            # Combine features to detect menu-like patterns
            uniform_ratio = uniform_regions / (regions_x * regions_y)
            is_menu_like = (uniform_ratio > 0.3 and edge_density > 0.05) or \
                          (uniform_ratio > 0.2 and has_aligned_elements)
            
            return is_menu_like
            
        except Exception as e:
            logger.error(f"Error in menu color detection: {e}")
            return False
        
    def _detect_menu_by_edges(self, frame_np: np.ndarray) -> bool:
        """Detect menu based on edge patterns typical of UI elements.
        
        Args:
            frame_np (np.ndarray): Current frame as numpy array
            
        Returns:
            bool: True if menu detected, False otherwise
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame_np, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Detect edges
        edges = cv2.Canny(blurred, 50, 150)
        
        # Look for horizontal and vertical lines (common in UI)
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=100, maxLineGap=10)
        
        # Check if sufficient lines were detected
        if lines is None:
            return False
            
        # Count horizontal and vertical lines
        horizontal_lines = 0
        vertical_lines = 0
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            
            # Calculate angle to determine horizontal or vertical
            if abs(x2 - x1) < 1:  # Avoid division by zero
                angle = 90.0
            else:
                angle = abs(np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi)
                
            # Classify lines
            if angle < 10:  # Nearly horizontal
                horizontal_lines += 1
            elif angle > 80:  # Nearly vertical
                vertical_lines += 1
                
        # UI typically has both horizontal and vertical elements
        if horizontal_lines >= 5 and vertical_lines >= 5:
            return True
            
        # Check if edges are concentrated in UI-typical regions
        edge_ratio = np.count_nonzero(edges) / edges.size
        
        # Menu UI typically has distinct edges
        if edge_ratio > 0.1:
            return True
            
        return False
        
    def _detect_resume_button(self, frame_np: np.ndarray) -> bool:
        """Detect resume button in the menu.
        
        Args:
            frame_np (np.ndarray): Current frame
            
        Returns:
            bool: True if resume button detected, False otherwise
        """
        if self.resume_button_template is None:
            return False
            
        # Convert to grayscale
        if len(frame_np.shape) == 3:
            gray_frame = cv2.cvtColor(frame_np, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = frame_np
            
        # Apply template matching
        result = cv2.matchTemplate(gray_frame, self.resume_button_template, cv2.TM_CCOEFF_NORMED)
        
        # Get maximum match value
        _, max_val, _, _ = cv2.minMaxLoc(result)
        
        # Threshold for detection
        return max_val > 0.6
    
    def save_current_frame_as_menu_reference(self, frame, save_path):
        """Save the current frame as a menu reference image.
        
        Args:
            frame: Current frame to save as reference
            save_path: Path to save the reference image
            
        Returns:
            bool: True if successfully saved, False otherwise
        """
        try:
            # Convert frame to proper format
            if isinstance(frame, torch.Tensor):
                frame_np = frame.detach().cpu().numpy()
                
                # Convert from PyTorch's CHW format to HWC
                if len(frame_np.shape) == 3 and frame_np.shape[0] == 3:
                    frame_np = frame_np.transpose(1, 2, 0)
                    
                # Scale to 0-255 if needed
                if frame_np.max() <= 1.0:
                    frame_np = (frame_np * 255).astype(np.uint8)
            else:
                frame_np = frame.astype(np.uint8) if frame.dtype != np.uint8 else frame
            
            # Convert to grayscale
            if len(frame_np.shape) == 3 and frame_np.shape[2] == 3:
                gray_frame = cv2.cvtColor(frame_np, cv2.COLOR_RGB2GRAY)
            else:
                gray_frame = frame_np
            
            # Save the image
            cv2.imwrite(save_path, gray_frame)
            
            # Update our reference
            self.menu_reference = gray_frame
            
            # Compute keypoints and descriptors
            self.menu_kp, self.menu_desc = self.menu_matcher.detectAndCompute(gray_frame, None)
            self.menu_detection_initialized = True
            
            logger.info(f"Successfully saved menu reference to {save_path} with {len(self.menu_kp)} keypoints")
            return True
        except Exception as e:
            logger.error(f"Failed to save menu reference: {e}")
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

    def get_visual_change_score(self, frame: np.ndarray) -> float:
        """Get visual change score for a frame.
        
        Args:
            frame: Input frame
            
        Returns:
            float: Visual change score (higher = more change)
        """
        try:
            if self.visual_change_analyzer is None:
                return 0.0
            
            # Calculate visual change score using the analyzer
            # The VisualChangeAnalyzer may compare with previous frames
            return self.visual_change_analyzer.get_change_score(frame)
        except Exception as e:
            logger.error(f"Error calculating visual change score: {e}")
            return 0.0

    def detect_ui_elements(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect UI elements in a frame.
        
        Args:
            frame: Input frame
            
        Returns:
            List[Tuple[int, int, int, int]]: List of UI element bounding boxes (x, y, w, h)
        """
        try:
            # Safety check for None or invalid frames
            if frame is None or (hasattr(frame, 'numel') and frame.numel() == 0):
                logger.warning("Invalid frame passed to detect_ui_elements")
                return []
                
            if isinstance(frame, np.ndarray) and frame.size == 0:
                logger.warning("Empty numpy array passed to detect_ui_elements")
                return []
                
            # Get image utils instance safety
            if not hasattr(self, 'image_utils'):
                from src.utils.image_utils import ImageUtils
                self.image_utils = ImageUtils()
            
            # Use ImageUtils to detect UI elements with robust error handling
            try:
                # Handle tensor conversion if needed
                if hasattr(frame, 'detach') and hasattr(frame, 'cpu') and hasattr(frame, 'numpy'):
                    # It's a PyTorch tensor
                    frame_np = frame.detach().cpu().numpy()
                    
                    # Handle different tensor layouts
                    if len(frame_np.shape) == 3:
                        if frame_np.shape[0] == 3:  # CHW format
                            frame_np = np.transpose(frame_np, (1, 2, 0))  # Convert to HWC
                elif isinstance(frame, np.ndarray):
                    frame_np = frame
                else:
                    logger.warning(f"Unsupported frame type in detect_ui_elements: {type(frame)}")
                    return []
                
                # Ensure proper data type
                if frame_np.dtype != np.uint8:
                    if np.max(frame_np) <= 1.0:
                        frame_np = (frame_np * 255).astype(np.uint8)
                    else:
                        frame_np = frame_np.astype(np.uint8)
                
                # Ensure frame has valid dimensions
                if frame_np.ndim < 2:
                    logger.warning(f"Invalid frame dimensions: {frame_np.shape}")
                    return []
                
                # Ensure frame is not empty
                if frame_np.size == 0 or 0 in frame_np.shape:
                    logger.warning(f"Empty frame dimensions: {frame_np.shape}")
                    return []
                
                # Use ImageUtils to detect UI elements
                standard_elements = self.image_utils.detect_ui_elements(frame_np)
                if standard_elements is None:
                    standard_elements = []
                
                return standard_elements
            except Exception as e:
                logger.error(f"Error in detect_ui_elements using ImageUtils: {e}")
                return []
        except Exception as e:
            logger.error(f"Uncaught error in detect_ui_elements: {e}")
            return []

    def calculate_ui_density(self, frame: np.ndarray) -> float:
        """Calculate UI element density from a frame.
        
        Args:
            frame: Input frame
            
        Returns:
            float: UI element density score (0.0 to 1.0)
        """
        try:
            # Detect UI elements
            ui_elements = self.detect_ui_elements(frame)
            
            if not ui_elements:
                return 0.0
            
            # Calculate frame dimensions
            if isinstance(frame, torch.Tensor):
                if frame.dim() == 3 and frame.shape[0] == 3:  # CHW format
                    height, width = frame.shape[1], frame.shape[2]
                else:
                    height, width = frame.shape[:2]
            else:  # Numpy array
                height, width = frame.shape[:2]
            
            # Calculate total area of UI elements and frame
            total_ui_area = sum(w * h for _, _, w, h in ui_elements)
            frame_area = height * width
            
            # Calculate density
            density = min(1.0, total_ui_area / (frame_area * 0.3))  # Normalize with a 30% threshold
            
            # Add bonus for well-distributed UI elements across the screen
            screen_regions = [0, 0, 0, 0]  # top-left, top-right, bottom-left, bottom-right
            for x, y, w, h in ui_elements:
                center_x, center_y = x + w//2, y + h//2
                
                # Determine which region this element is in
                region_idx = (center_y > height//2) * 2 + (center_x > width//2)
                screen_regions[region_idx] += 1
            
            # Bonus for UI elements in multiple regions (likely menu)
            region_coverage = len([r for r in screen_regions if r > 0]) / 4.0
            density += region_coverage * 0.2  # Add up to 0.2 for full region coverage
            
            return min(1.0, density)
        except Exception as e:
            logger.error(f"Error calculating UI density: {e}")
            return 0.0

    def detect_menu_contrast_pattern(self, frame: np.ndarray) -> float:
        """Detect menu-like contrast patterns in a frame.
        
        Args:
            frame: Input frame
            
        Returns:
            float: Menu pattern confidence (0.0 to 1.0)
        """
        try:
            # Convert PyTorch tensor to numpy if needed
            if isinstance(frame, torch.Tensor):
                frame_np = frame.detach().cpu().numpy()
                # Convert from PyTorch's CHW format to HWC format if needed
                if len(frame_np.shape) == 3 and frame_np.shape[0] == 3:
                    frame_np = frame_np.transpose(1, 2, 0)
                # Ensure values are in range 0-255
                if frame_np.max() <= 1.0:
                    frame_np = (frame_np * 255).astype(np.uint8)
            else:
                frame_np = frame
            
            # Convert to grayscale for analysis
            if len(frame_np.shape) == 3:
                gray = cv2.cvtColor(frame_np, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame_np
            
            # 1. Analyze edge intensity in central region (menus often have strong edges in center)
            height, width = gray.shape[:2]
            center_x, center_y = width // 2, height // 2
            center_width, center_height = width // 2, height // 2
            center_region = gray[
                max(0, center_y - center_height//2):min(height, center_y + center_height//2),
                max(0, center_x - center_width//2):min(width, center_x + center_width//2)
            ]
            
            # Calculate edge intensity
            center_edges = cv2.Canny(center_region, 30, 150)
            edge_intensity = np.mean(center_edges) / 255.0
            
            # 2. Check for contrast variation (menus often have consistent contrast)
            # Calculate local contrast using standard deviation of pixel values
            local_contrasts = []
            for y in range(0, height, height//4):
                for x in range(0, width, width//4):
                    region = gray[y:min(y+height//4, height), x:min(x+width//4, width)]
                    if region.size > 0:
                        local_contrasts.append(np.std(region))
            
            # Lower variance in contrast is more menu-like
            contrast_variance = np.std(local_contrasts) / np.mean(local_contrasts) if np.mean(local_contrasts) > 0 else 1.0
            contrast_score = max(0, 1.0 - contrast_variance)
            
            # 3. Check for horizontal/vertical line presence (menu borders)
            horizontal_lines = cv2.HoughLinesP(
                center_edges, 1, np.pi/180, threshold=20, 
                minLineLength=width//10, maxLineGap=width//20
            )
            vertical_lines = cv2.HoughLinesP(
                center_edges, 1, np.pi/180, threshold=20, 
                minLineLength=height//10, maxLineGap=height//20
            )
            
            line_score = 0.0
            if horizontal_lines is not None or vertical_lines is not None:
                h_count = 0 if horizontal_lines is None else len(horizontal_lines)
                v_count = 0 if vertical_lines is None else len(vertical_lines)
                line_score = min(1.0, (h_count + v_count) / 10.0)  # Normalize, max of 10 lines
            
            # Combine scores with weights
            menu_score = (edge_intensity * 0.3 + contrast_score * 0.3 + line_score * 0.4)
            
            return menu_score
        except Exception as e:
            logger.error(f"Error detecting menu contrast pattern: {e}")
            return 0.0

    def calculate_frame_difference(self, frame1, frame2):
        """Calculate the difference between two frames.
        
        Args:
            frame1: First frame (torch.Tensor or numpy array)
            frame2: Second frame (torch.Tensor or numpy array)
            
        Returns:
            float: Difference score between 0 and 1
        """
        # Ensure frames are valid
        if frame1 is None or frame2 is None:
            return 0.0
            
        # Convert to numpy if needed
        if isinstance(frame1, torch.Tensor):
            frame1_np = frame1.detach().cpu().numpy()
            # Convert CHW to HWC format if needed
            if len(frame1_np.shape) == 3 and frame1_np.shape[0] == 3:
                frame1_np = np.transpose(frame1_np, (1, 2, 0))
        else:
            frame1_np = frame1
            
        if isinstance(frame2, torch.Tensor):
            frame2_np = frame2.detach().cpu().numpy()
            # Convert CHW to HWC format if needed
            if len(frame2_np.shape) == 3 and frame2_np.shape[0] == 3:
                frame2_np = np.transpose(frame2_np, (1, 2, 0))
        else:
            frame2_np = frame2
            
        # Convert to grayscale if needed
        if len(frame1_np.shape) == 3 and frame1_np.shape[2] == 3:
            frame1_gray = cv2.cvtColor(frame1_np.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            frame1_gray = frame1_np.astype(np.uint8)
            
        if len(frame2_np.shape) == 3 and frame2_np.shape[2] == 3:
            frame2_gray = cv2.cvtColor(frame2_np.astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            frame2_gray = frame2_np.astype(np.uint8)
            
        # Ensure frames are the same size
        if frame1_gray.shape != frame2_gray.shape:
            # Resize frame2 to match frame1
            frame2_gray = cv2.resize(frame2_gray, (frame1_gray.shape[1], frame1_gray.shape[0]))
            
        # Calculate mean absolute difference
        diff = cv2.absdiff(frame1_gray, frame2_gray)
        diff_score = np.mean(diff) / 255.0
        
        return diff_score 

    def detect_version_string(self, frame: np.ndarray) -> bool:
        """Detect version string in the bottom-left corner of the screen.
        This is a reliable indicator for menu screens.
        
        Args:
            frame: Input frame
            
        Returns:
            bool: True if version string detected
        """
        try:
            # First ensure frame is valid
            if frame is None or frame.size == 0:
                return False
            
            # Convert float32 (0-1) to uint8 (0-255) if needed
            if frame.dtype == np.float32:
                frame = (frame * 255).astype(np.uint8)
            
            # Ensure frame has proper shape before processing
            if len(frame.shape) != 3 or frame.shape[2] != 3:
                if len(frame.shape) == 3:
                    # Reshape incorrectly shaped 3D arrays
                    height, width, depth = frame.shape
                    if depth != 3:
                        if height == 3:  # Likely a CHW format
                            frame = frame.transpose(1, 2, 0)
                        else:
                            # Can't process this frame properly
                            return False
                else:
                    # Can't process this frame properly
                    return False
            
            # Extract bottom-left region (where version string usually appears)
            height, width = frame.shape[:2]
            bottom_left = frame[height-30:height, 0:int(width*0.3)]
            
            # Convert to grayscale
            if bottom_left.shape[2] >= 3:
                gray = cv2.cvtColor(bottom_left, cv2.COLOR_RGB2GRAY)
            else:
                gray = bottom_left
            
            # Apply threshold to find text
            _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)
            
            # Check for characteristic patterns of text
            # Version string typically has high contrast ratio in small region
            white_ratio = np.sum(thresh > 200) / thresh.size
            return white_ratio > 0.01 and white_ratio < 0.2  # Version strings have some white text but not too much
        except Exception as e:
            logger.error(f"Error detecting version string: {e}")
            return False 