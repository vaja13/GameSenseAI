import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from pathlib import Path
import logging

class GameplayAnalysis:
    def __init__(self, full_map_path, yolo_model_path):
        """
        Initialize the gameplay analysis pipeline
        
        Args:
            full_map_path (str): Path to the full map image
            yolo_model_path (str): Path to the trained YOLO model weights
        """
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize SIFT and FLANN matcher for homography
        self.sift = cv2.SIFT_create()
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        self.matcher = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Load the full map
        self.load_full_map(full_map_path)
        
        # Load YOLO model
        self.load_yolo_model(yolo_model_path)
        
        # Initialize homography matrix
        self.homography_matrix = None
        
        # Minimap region coordinates (x, y, width, height)
        self.minimap_region = {
            'x': 15,      # Top-left x coordinate
            'y': 15,      # Top-left y coordinate
            'width': 430, # Width of minimap region
            'height': 435 # Height of minimap region
        }
        
    def extract_minimap(self, frame):
        """Extract minimap region from the gameplay frame"""
        try:
            if frame is None:
                raise ValueError("Input frame is None")
                
            height, width = frame.shape[:2]
            
            # Calculate bottom-right coordinates
            x2 = self.minimap_region['x'] + self.minimap_region['width']
            y2 = self.minimap_region['y'] + self.minimap_region['height']
            
            # Validate minimap coordinates
            if (x2 > width or 
                y2 > height or
                self.minimap_region['x'] < 0 or 
                self.minimap_region['y'] < 0):
                raise ValueError("Invalid minimap coordinates for frame size")
                
            # Crop minimap region
            minimap = frame[
                self.minimap_region['y']:y2,
                self.minimap_region['x']:x2
            ]
            
            return minimap
        except Exception as e:
            self.logger.error(f"Error extracting minimap: {str(e)}")
            return None
        
    def load_full_map(self, map_path):
        """Load the full map image"""
        try:
            self.full_map = cv2.imread(map_path)
            if self.full_map is None:
                raise ValueError("Failed to load full map")
            self.full_map_gray = cv2.cvtColor(self.full_map, cv2.COLOR_BGR2GRAY)
            self.logger.info("Full map loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading full map: {str(e)}")
            raise
            
    def load_yolo_model(self, model_path):
        """Load the YOLO model"""
        try:
            if not Path(model_path).exists():
                raise FileNotFoundError(f"YOLO model not found at {model_path}")
            self.yolo_model = YOLO(model_path)
            self.logger.info("YOLO model loaded successfully")
        except Exception as e:
            self.logger.error(f"Error loading YOLO model: {str(e)}")
            raise
    
            
    def detect_players(self, minimap):
        """Detect players in the minimap using YOLO"""
        try:
            if minimap is None:
                raise ValueError("Input minimap is None")
                
            # Run YOLO detection
            results = self.yolo_model(minimap, conf=0.25)  # Adjust confidence threshold as needed
            
            # Extract center points of detected bounding boxes
            centers = []
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get box coordinates
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    # Calculate center point
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    centers.append((center_x, center_y))
            
            return centers
        except Exception as e:
            self.logger.error(f"Error detecting players: {str(e)}")
            return []
            
    def compute_homography(self, minimap):
        """Compute homography matrix between minimap and full map"""
        try:
            if minimap is None:
                raise ValueError("Input minimap is None")
                
            # Convert minimap to grayscale
            minimap_gray = cv2.cvtColor(minimap, cv2.COLOR_BGR2GRAY)
            
            # Detect keypoints and descriptors
            kp_minimap, des_minimap = self.sift.detectAndCompute(minimap_gray, None)
            kp_full_map, des_full_map = self.sift.detectAndCompute(self.full_map_gray, None)
            
            if des_minimap is None or des_full_map is None:
                raise ValueError("Feature detection failed")
                
            # Find matches
            matches = self.matcher.knnMatch(des_minimap, des_full_map, k=2)
            
            # Apply Lowe's ratio test
            good_matches = []
            for m, n in matches:
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
                    
            if len(good_matches) < 4:
                raise ValueError("Not enough good matches for homography")
                
            # Extract matched keypoints
            src_pts = np.float32([kp_minimap[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp_full_map[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            
            # Compute homography
            self.homography_matrix, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            
            return True
        except Exception as e:
            self.logger.error(f"Error computing homography: {str(e)}")
            return False
            
    def transform_points(self, points):
        """Transform points from minimap to full map coordinates"""
        try:
            if self.homography_matrix is None:
                raise ValueError("Homography matrix not computed")
                
            transformed_points = []
            for x, y in points:
                # Convert to homogeneous coordinates
                point = np.array([[x, y, 1]], dtype=np.float32).T
                
                # Apply transformation
                transformed = np.dot(self.homography_matrix, point)
                
                # Normalize coordinates
                x_main = int(transformed[0] / transformed[2])
                y_main = int(transformed[1] / transformed[2])
                
                transformed_points.append((x_main, y_main))
                
            return transformed_points
        except Exception as e:
            self.logger.error(f"Error transforming points: {str(e)}")
            return []
            
    def visualize_result(self, transformed_points):
        """Create visualization of players on the full map"""
        try:
            if not transformed_points:
                raise ValueError("No points to visualize")
                
            result_map = self.full_map.copy()
            
            # Draw players on the map
            for x, y in transformed_points:
                # Draw player marker (red circle)
                cv2.circle(result_map, (x, y), radius=8, color=(0, 0, 255), thickness=-1)
                # Draw outline
                cv2.circle(result_map, (x, y), radius=8, color=(255, 255, 255), thickness=2)
                
            return result_map
        except Exception as e:
            self.logger.error(f"Error creating visualization: {str(e)}")
            return None
            
    def process_frame(self, frame):
        """Process a single gameplay frame"""
        try:
            # Extract minimap region
            minimap = self.extract_minimap(frame)
            if minimap is None:
                raise ValueError("Failed to extract minimap")
                
            # Detect players
            player_centers = self.detect_players(minimap)
            if not player_centers:
                raise ValueError("No players detected")
                
            # Compute homography if not already computed
            if self.homography_matrix is None:
                if not self.compute_homography(minimap):
                    raise ValueError("Failed to compute homography")
                
            # Transform player positions
            transformed_points = self.transform_points(player_centers)
            if not transformed_points:
                raise ValueError("Failed to transform points")
                
            # Create visualization
            result = self.visualize_result(transformed_points)
            if result is None:
                raise ValueError("Failed to create visualization")
                
            return result
        except Exception as e:
            self.logger.error(f"Error processing frame: {str(e)}")
            return None