"""
Menu templates for Cities: Skylines 2 environment.

This module provides functionality for managing menu templates for detection.
"""

import logging
import os
import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class MenuTemplateManager:
    """Manages templates for menu detection."""
    
    def __init__(
        self,
        templates_dir: str = "menu_templates",
        metadata_file: str = "menu_metadata.json"
    ):
        """Initialize menu template manager.
        
        Args:
            templates_dir: Directory to store templates
            metadata_file: File to store template metadata
        """
        self.templates_dir = Path(templates_dir)
        self.metadata_file = self.templates_dir / metadata_file
        self.templates = {}
        self.metadata = {}
        
        # Ensure templates directory exists
        os.makedirs(self.templates_dir, exist_ok=True)
        
        # Load existing templates and metadata
        self._load_templates()
        self._load_metadata()
        
        logger.info(f"Menu template manager initialized with {len(self.templates)} templates")
    
    def _load_templates(self):
        """Load template images from disk."""
        if not self.templates_dir.exists():
            logger.warning(f"Templates directory {self.templates_dir} does not exist")
            return
            
        try:
            # Find all image files in the templates directory
            image_files = list(self.templates_dir.glob("*.png")) + list(self.templates_dir.glob("*.jpg"))
            
            for image_file in image_files:
                try:
                    # Extract menu type from filename
                    menu_type = image_file.stem
                    
                    # Load the image
                    template = cv2.imread(str(image_file))
                    
                    if template is not None:
                        self.templates[menu_type] = template
                        logger.info(f"Loaded template for menu type: {menu_type}")
                    else:
                        logger.warning(f"Failed to load template image: {image_file}")
                        
                except Exception as e:
                    logger.error(f"Error loading template {image_file}: {e}")
            
        except Exception as e:
            logger.error(f"Error loading templates: {e}")
    
    def _load_metadata(self):
        """Load template metadata from disk."""
        if not self.metadata_file.exists():
            logger.warning(f"Metadata file {self.metadata_file} does not exist")
            return
            
        try:
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
                logger.info(f"Loaded metadata for {len(self.metadata)} templates")
                
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            # Initialize empty metadata if file couldn't be loaded
            self.metadata = {}
    
    def _save_metadata(self):
        """Save template metadata to disk."""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
                logger.info(f"Saved metadata for {len(self.metadata)} templates")
                
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    def add_template(
        self,
        menu_type: str,
        template_image: np.ndarray,
        signature_regions: Optional[List[Tuple[float, float, float, float]]] = None,
        threshold: float = 0.7
    ) -> bool:
        """Add a new template for a menu type.
        
        Args:
            menu_type: Type of menu
            template_image: Image of the menu
            signature_regions: Regions that are characteristic of this menu
            threshold: Detection threshold
            
        Returns:
            Whether the template was added successfully
        """
        try:
            # Save the template image
            template_path = self.templates_dir / f"{menu_type}.png"
            cv2.imwrite(str(template_path), template_image)
            
            # Add to in-memory templates
            self.templates[menu_type] = template_image
            
            # Update metadata
            self.metadata[menu_type] = {
                "path": str(template_path),
                "threshold": threshold,
                "signature_regions": signature_regions or [],
                "timestamp": str(Path(template_path).stat().st_mtime)
            }
            
            # Save metadata
            self._save_metadata()
            
            logger.info(f"Added template for menu type: {menu_type}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding template for {menu_type}: {e}")
            return False
    
    def get_template(self, menu_type: str) -> Optional[np.ndarray]:
        """Get a template for a menu type.
        
        Args:
            menu_type: Type of menu
            
        Returns:
            Template image or None if not found
        """
        return self.templates.get(menu_type)
    
    def get_all_templates(self) -> Dict[str, np.ndarray]:
        """Get all templates.
        
        Returns:
            Dictionary of menu types to template images
        """
        return self.templates.copy()
    
    def get_metadata(self, menu_type: str) -> Optional[Dict[str, Any]]:
        """Get metadata for a menu type.
        
        Args:
            menu_type: Type of menu
            
        Returns:
            Metadata dictionary or None if not found
        """
        return self.metadata.get(menu_type)
    
    def get_all_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Get all metadata.
        
        Returns:
            Dictionary of menu types to metadata
        """
        return self.metadata.copy()
    
    def remove_template(self, menu_type: str) -> bool:
        """Remove a template.
        
        Args:
            menu_type: Type of menu
            
        Returns:
            Whether the template was removed successfully
        """
        try:
            # Remove from in-memory templates
            if menu_type in self.templates:
                del self.templates[menu_type]
                
            # Remove from metadata
            if menu_type in self.metadata:
                template_path = self.metadata[menu_type].get("path")
                del self.metadata[menu_type]
                
                # Save metadata
                self._save_metadata()
                
                # Delete the file if it exists
                if template_path and os.path.exists(template_path):
                    os.remove(template_path)
                    
            logger.info(f"Removed template for menu type: {menu_type}")
            return True
            
        except Exception as e:
            logger.error(f"Error removing template for {menu_type}: {e}")
            return False
    
    def update_template(
        self,
        menu_type: str,
        template_image: Optional[np.ndarray] = None,
        signature_regions: Optional[List[Tuple[float, float, float, float]]] = None,
        threshold: Optional[float] = None
    ) -> bool:
        """Update an existing template.
        
        Args:
            menu_type: Type of menu
            template_image: New image of the menu
            signature_regions: New signature regions
            threshold: New detection threshold
            
        Returns:
            Whether the template was updated successfully
        """
        # Check if template exists
        if menu_type not in self.metadata:
            logger.warning(f"Template for menu type {menu_type} does not exist")
            return False
            
        try:
            # Update template image if provided
            if template_image is not None:
                template_path = self.templates_dir / f"{menu_type}.png"
                cv2.imwrite(str(template_path), template_image)
                self.templates[menu_type] = template_image
                self.metadata[menu_type]["timestamp"] = str(Path(template_path).stat().st_mtime)
            
            # Update signature regions if provided
            if signature_regions is not None:
                self.metadata[menu_type]["signature_regions"] = signature_regions
                
            # Update threshold if provided
            if threshold is not None:
                self.metadata[menu_type]["threshold"] = threshold
                
            # Save metadata
            self._save_metadata()
            
            logger.info(f"Updated template for menu type: {menu_type}")
            return True
            
        except Exception as e:
            logger.error(f"Error updating template for {menu_type}: {e}")
            return False
    
    def extract_template_from_frame(
        self,
        frame: np.ndarray,
        menu_type: str,
        region: Optional[Tuple[float, float, float, float]] = None
    ) -> Optional[np.ndarray]:
        """Extract a template from a frame.
        
        Args:
            frame: Frame to extract template from
            menu_type: Type of menu
            region: Region to extract (normalized coordinates)
            
        Returns:
            Extracted template or None if extraction failed
        """
        try:
            # If region is provided, extract it
            if region is not None:
                x1, y1, x2, y2 = region
                h, w = frame.shape[:2]
                
                # Convert normalized coordinates to pixel values
                x1_px, y1_px = int(x1 * w), int(y1 * h)
                x2_px, y2_px = int(x2 * w), int(y2 * h)
                
                # Extract the region
                template = frame[y1_px:y2_px, x1_px:x2_px]
            else:
                # Use the whole frame
                template = frame.copy()
                
            logger.info(f"Extracted template for menu type: {menu_type}")
            return template
            
        except Exception as e:
            logger.error(f"Error extracting template for {menu_type}: {e}")
            return None
    
    def learn_from_frame(
        self,
        frame: np.ndarray,
        menu_type: str,
        region: Optional[Tuple[float, float, float, float]] = None,
        threshold: float = 0.7,
        signature_regions: Optional[List[Tuple[float, float, float, float]]] = None
    ) -> bool:
        """Learn a new template from a frame.
        
        Args:
            frame: Frame to learn from
            menu_type: Type of menu
            region: Region to extract (normalized coordinates)
            threshold: Detection threshold
            signature_regions: Regions that are characteristic of this menu
            
        Returns:
            Whether the template was learned successfully
        """
        # Extract template
        template = self.extract_template_from_frame(frame, menu_type, region)
        if template is None:
            return False
            
        # Add the template
        if menu_type in self.templates:
            # Update existing template
            return self.update_template(menu_type, template, signature_regions, threshold)
        else:
            # Add new template
            return self.add_template(menu_type, template, signature_regions, threshold) 