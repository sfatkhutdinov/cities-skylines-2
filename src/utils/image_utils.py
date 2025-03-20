import numpy as np
import cv2
from typing import List, Tuple, Optional, Union, Dict
import logging

logger = logging.getLogger(__name__)

class ImageUtils:
    """Utilities for image processing and analysis."""
    
    def __init__(self, debug_mode: bool = False):
        """Initialize image utilities.
        
        Args:
            debug_mode: Whether to enable debug visualizations
        """
        self.debug_mode = debug_mode
        self.last_processed_frame = None
        self.last_frame_hash = None
        
    def preprocess_frame(self, frame: np.ndarray, 
                         resolution: Optional[Tuple[int, int]] = None,
                         normalize: bool = True,
                         grayscale: bool = False) -> np.ndarray:
        """Preprocess a frame for analysis.
        
        Args:
            frame: Input frame
            resolution: Target resolution (height, width) or None to keep original
            normalize: Whether to normalize pixel values to [0, 1]
            grayscale: Whether to convert to grayscale
            
        Returns:
            np.ndarray: Preprocessed frame
        """
        if frame is None:
            logger.warning("Received None frame for preprocessing")
            return np.zeros((240, 320, 3), dtype=np.uint8)
            
        # Check frame validity
        if not isinstance(frame, np.ndarray):
            logger.error(f"Invalid frame type: {type(frame)}")
            return np.zeros((240, 320, 3), dtype=np.uint8)
            
        if frame.size == 0 or frame.ndim < 2:
            logger.error(f"Invalid frame shape: {frame.shape}")
            return np.zeros((240, 320, 3), dtype=np.uint8)
        
        # Make a copy to avoid modifying the original
        processed = frame.copy()
        
        # Convert to grayscale if requested
        if grayscale and processed.ndim == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
            # Add channel dimension for consistency
            processed = np.expand_dims(processed, axis=-1)
        
        # Resize if resolution specified
        if resolution is not None:
            try:
                if processed.ndim == 3:
                    processed = cv2.resize(processed, (resolution[1], resolution[0]), 
                                          interpolation=cv2.INTER_AREA)
                else:
                    processed = cv2.resize(processed, (resolution[1], resolution[0]), 
                                          interpolation=cv2.INTER_AREA)
                    processed = np.expand_dims(processed, axis=-1)
            except Exception as e:
                logger.error(f"Resize failed: {e}")
        
        # Normalize to [0, 1] if requested
        if normalize:
            processed = processed.astype(np.float32) / 255.0
        
        self.last_processed_frame = processed
        return processed
    
    def detect_edges(self, frame: np.ndarray, 
                    low_threshold: int = 50, 
                    high_threshold: int = 150) -> np.ndarray:
        """Detect edges in a frame using Canny edge detection.
        
        Args:
            frame: Input frame
            low_threshold: Lower threshold for edge detection
            high_threshold: Higher threshold for edge detection
            
        Returns:
            np.ndarray: Edge detection result
        """
        if frame is None:
            return np.zeros((240, 320), dtype=np.uint8)
            
        # Convert to grayscale if needed
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, low_threshold, high_threshold)
        
        return edges
    
    def detect_ui_elements(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect UI elements in a frame.
        
        Args:
            frame: Input frame
            
        Returns:
            List[Tuple[int, int, int, int]]: List of UI element bounding boxes (x, y, w, h)
        """
        try:
            if frame is None:
                logger.warning("Received None frame in detect_ui_elements")
                return []
                
            # Check frame validity
            if not isinstance(frame, np.ndarray):
                logger.error(f"Invalid frame type in detect_ui_elements: {type(frame)}")
                return []
                
            if frame.size == 0 or frame.ndim < 2:
                logger.error(f"Invalid frame shape in detect_ui_elements: {frame.shape if hasattr(frame, 'shape') else 'unknown'}")
                return []
                
            # Ensure frame is in the correct format (HWC or grayscale)
            if len(frame.shape) == 3 and frame.shape[0] == 3 and len(frame.shape) == 3:
                # Convert from CHW to HWC
                frame = np.transpose(frame, (1, 2, 0))
                
            # Ensure correct data type
            if frame.dtype != np.uint8:
                if np.max(frame) <= 1.0:
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
                    
            # Convert to grayscale if needed
            try:
                if frame.ndim == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    gray = frame
                    
                # Apply thresholding to highlight UI elements (buttons, text, etc.)
                _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
                
                # Find contours in the thresholded image
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                # Check if contours is None (shouldn't happen but being defensive)
                if contours is None:
                    logger.warning("No contours found in detect_ui_elements")
                    return []
                
                # Filter contours by size to find likely UI elements
                ui_elements = []
                for contour in contours:
                    try:
                        x, y, w, h = cv2.boundingRect(contour)
                        
                        # Filter by size to exclude noise
                        if w > 10 and h > 10 and w < frame.shape[1] // 2 and h < frame.shape[0] // 2:
                            ui_elements.append((x, y, w, h))
                    except Exception as e:
                        logger.warning(f"Error processing contour: {e}")
                        continue
                        
                return ui_elements
            except cv2.error as e:
                logger.error(f"OpenCV error in detect_ui_elements: {e}")
                return []
            except Exception as e:
                logger.error(f"Error in image processing for detect_ui_elements: {e}")
                return []
        except Exception as e:
            logger.error(f"Uncaught exception in detect_ui_elements: {e}")
            return []
    
    def template_match(self, frame: np.ndarray, template: np.ndarray, 
                      threshold: float = 0.8, 
                      method: int = cv2.TM_CCOEFF_NORMED) -> Tuple[float, Optional[Tuple[int, int, int, int]]]:
        """Match a template in a frame.
        
        Args:
            frame: Input frame
            template: Template to match
            threshold: Matching threshold
            method: OpenCV template matching method
            
        Returns:
            Tuple[float, Optional[Tuple[int, int, int, int]]]: 
                - Maximum match score
                - Bounding box of the match (x, y, w, h) or None if no match
        """
        if frame is None or template is None:
            return 0.0, None
            
        # Check template is smaller than frame
        if (template.shape[0] > frame.shape[0] or 
            template.shape[1] > frame.shape[1]):
            logger.warning("Template larger than frame, skipping match")
            return 0.0, None
            
        # Convert to grayscale if needed
        if frame.ndim == 3 and template.ndim == 3:
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
        else:
            frame_gray = frame.copy()
            template_gray = template.copy()
        
        # Perform template matching
        try:
            result = cv2.matchTemplate(frame_gray, template_gray, method)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            
            # Use max_val for TM_CCOEFF* methods, 1-min_val for TM_SQDIFF* methods
            if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
                match_score = 1.0 - min_val
                match_loc = min_loc
            else:
                match_score = max_val
                match_loc = max_loc
                
            if match_score >= threshold:
                w, h = template_gray.shape[1], template_gray.shape[0]
                match_rect = (match_loc[0], match_loc[1], w, h)
                return match_score, match_rect
            else:
                return match_score, None
                
        except Exception as e:
            logger.error(f"Template matching failed: {e}")
            return 0.0, None
    
    def find_text_regions(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Find regions likely to contain text.
        
        Args:
            frame: Input frame
            
        Returns:
            List[Tuple[int, int, int, int]]: List of text region bounding boxes (x, y, w, h)
        """
        if frame is None:
            return []
            
        # Convert to grayscale if needed
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            
        # Apply MSER (Maximally Stable Extremal Regions) for text detection
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)
        
        # Convert regions to bounding boxes
        text_regions = []
        for region in regions:
            x, y, w, h = cv2.boundingRect(region)
            
            # Filter by aspect ratio and size to find text-like regions
            aspect_ratio = w / h
            if 0.2 < aspect_ratio < 10 and w > 10 and h > 5:
                text_regions.append((x, y, w, h))
                
        # Merge overlapping boxes
        text_regions = self._merge_overlapping_boxes(text_regions)
        
        return text_regions
    
    def _merge_overlapping_boxes(self, boxes: List[Tuple[int, int, int, int]], 
                               overlap_threshold: float = 0.3) -> List[Tuple[int, int, int, int]]:
        """Merge overlapping bounding boxes.
        
        Args:
            boxes: List of bounding boxes (x, y, w, h)
            overlap_threshold: IoU threshold for merging
            
        Returns:
            List[Tuple[int, int, int, int]]: Merged bounding boxes
        """
        if not boxes:
            return []
            
        # Sort by x coordinate
        boxes = sorted(boxes, key=lambda box: box[0])
        
        merged_boxes = []
        while boxes:
            # Take the first box
            current = boxes.pop(0)
            
            # Check for overlap with all remaining boxes
            i = 0
            while i < len(boxes):
                # Calculate intersection over union (IoU)
                iou = self._calculate_iou(current, boxes[i])
                
                if iou > overlap_threshold:
                    # Merge the boxes
                    x1 = min(current[0], boxes[i][0])
                    y1 = min(current[1], boxes[i][1])
                    x2 = max(current[0] + current[2], boxes[i][0] + boxes[i][2])
                    y2 = max(current[1] + current[3], boxes[i][1] + boxes[i][3])
                    
                    current = (x1, y1, x2 - x1, y2 - y1)
                    
                    # Remove the merged box
                    boxes.pop(i)
                else:
                    i += 1
                    
            merged_boxes.append(current)
            
        return merged_boxes
    
    def _calculate_iou(self, box1: Tuple[int, int, int, int], 
                     box2: Tuple[int, int, int, int]) -> float:
        """Calculate intersection over union between two boxes.
        
        Args:
            box1: First box (x, y, w, h)
            box2: Second box (x, y, w, h)
            
        Returns:
            float: IoU score
        """
        # Convert to (x1, y1, x2, y2) format
        box1_x1, box1_y1 = box1[0], box1[1]
        box1_x2, box1_y2 = box1[0] + box1[2], box1[1] + box1[3]
        
        box2_x1, box2_y1 = box2[0], box2[1]
        box2_x2, box2_y2 = box2[0] + box2[2], box2[1] + box2[3]
        
        # Calculate intersection area
        x_left = max(box1_x1, box2_x1)
        y_top = max(box1_y1, box2_y1)
        x_right = min(box1_x2, box2_x2)
        y_bottom = min(box1_y2, box2_y2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
        box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
        union_area = box1_area + box2_area - intersection_area
        
        if union_area == 0:
            return 0.0
            
        return intersection_area / union_area
    
    def frame_difference(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate the difference between two frames.
        
        Args:
            frame1: First frame
            frame2: Second frame
            
        Returns:
            float: Difference score (0.0 to 1.0, where 0.0 means identical frames)
        """
        if frame1 is None or frame2 is None:
            return 1.0
            
        # Make sure frames have the same shape
        if frame1.shape != frame2.shape:
            frame2 = cv2.resize(frame2, (frame1.shape[1], frame1.shape[0]))
            
        # Convert to grayscale if needed
        if frame1.ndim == 3:
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        else:
            gray1 = frame1
            
        if frame2.ndim == 3:
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        else:
            gray2 = frame2
            
        # Calculate mean absolute difference
        diff = cv2.absdiff(gray1, gray2)
        diff_score = np.mean(diff) / 255.0
        
        return diff_score
    
    def detect_motion(self, prev_frame: np.ndarray, curr_frame: np.ndarray, 
                     threshold: float = 25.0) -> Tuple[bool, np.ndarray, float]:
        """Detect motion between frames.
        
        Args:
            prev_frame: Previous frame
            curr_frame: Current frame
            threshold: Motion detection threshold
            
        Returns:
            Tuple[bool, np.ndarray, float]:
                - Whether motion was detected
                - Motion mask
                - Motion score
        """
        if prev_frame is None or curr_frame is None:
            return False, np.zeros((240, 320), dtype=np.uint8), 0.0
            
        # Make sure frames have the same shape
        if prev_frame.shape != curr_frame.shape:
            curr_frame = cv2.resize(curr_frame, (prev_frame.shape[1], prev_frame.shape[0]))
            
        # Convert to grayscale
        if prev_frame.ndim == 3:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        else:
            prev_gray = prev_frame
            
        if curr_frame.ndim == 3:
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        else:
            curr_gray = curr_frame
            
        # Calculate absolute difference
        frame_diff = cv2.absdiff(prev_gray, curr_gray)
        
        # Apply threshold to get motion mask
        _, motion_mask = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        motion_mask = cv2.dilate(motion_mask, kernel, iterations=2)
        
        # Calculate motion score
        motion_score = np.mean(motion_mask)
        
        # Detect if motion exceeds threshold
        motion_detected = motion_score > threshold
        
        return motion_detected, motion_mask, motion_score
    
    def compute_frame_hash(self, frame: np.ndarray, hash_size: int = 16) -> np.ndarray:
        """Compute a perceptual hash for a frame.
        
        Args:
            frame: Input frame
            hash_size: Size of the perceptual hash
            
        Returns:
            np.ndarray: Binary hash
        """
        if frame is None:
            return np.zeros((hash_size, hash_size), dtype=np.bool_)
            
        # Convert to grayscale
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            
        # Resize to hash_size x hash_size
        resized = cv2.resize(gray, (hash_size + 1, hash_size))
        
        # Compute difference hash
        diff = resized[:, 1:] > resized[:, :-1]
        
        return diff
    
    def hash_distance(self, hash1: np.ndarray, hash2: np.ndarray) -> float:
        """Calculate normalized Hamming distance between two image hashes.
        
        Args:
            hash1: First hash
            hash2: Second hash
            
        Returns:
            float: Normalized Hamming distance (0.0 to 1.0)
        """
        if hash1 is None or hash2 is None:
            return 1.0
            
        # Compute Hamming distance
        distance = np.count_nonzero(hash1 != hash2)
        
        # Normalize by hash size
        return distance / float(hash1.size)
    
    def is_duplicate_frame(self, frame: np.ndarray, 
                          threshold: float = 0.1) -> bool:
        """Check if a frame is a duplicate of the last processed frame.
        
        Args:
            frame: Input frame
            threshold: Hash distance threshold
            
        Returns:
            bool: True if frame is a duplicate, False otherwise
        """
        if frame is None or self.last_processed_frame is None:
            return False
            
        # Compute hash for current frame
        current_hash = self.compute_frame_hash(frame)
        
        if self.last_frame_hash is None:
            self.last_frame_hash = current_hash
            return False
            
        # Calculate hash distance
        distance = self.hash_distance(current_hash, self.last_frame_hash)
        
        # Update last frame hash
        self.last_frame_hash = current_hash
        
        # Consider duplicate if distance below threshold
        return distance < threshold 