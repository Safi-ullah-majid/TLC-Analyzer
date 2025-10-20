import torch
import cv2
import numpy as np
from ultralytics import YOLO
import json

class TLCAnalyzer:
    def __init__(self, yolo_model_path, recommendation_model_path):
        """Initialize TLC Analyzer with trained models"""
        self.yolo_model = YOLO(yolo_model_path)
        
        # Load recommendation model
        checkpoint = torch.load(recommendation_model_path, map_location='cpu')
        self.scaler = checkpoint['scaler']
        
        # Define and load recommendation model
        from torch import nn
        
        class RecommendationModel(nn.Module):
            def __init__(self, input_size=8):
                super(RecommendationModel, self).__init__()
                self.shared = nn.Sequential(
                    nn.Linear(input_size, 64), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2)
                )
                self.solvent_head = nn.Sequential(
                    nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 3)
                )
                self.columns_head = nn.Sequential(
                    nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1)
                )
                self.time_head = nn.Sequential(
                    nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1)
                )
            
            def forward(self, x):
                shared = self.shared(x)
                return (self.solvent_head(shared), 
                       self.columns_head(shared), 
                       self.time_head(shared))
        
        self.rec_model = RecommendationModel(input_size=checkpoint['input_size'])
        self.rec_model.load_state_dict(checkpoint['model_state_dict'])
        self.rec_model.eval()
        
        print("✓ Models loaded successfully!")
    
    def analyze_tlc(self, image_path):
        """Analyze TLC plate image and return recommendations"""
        
        # Step 1: Detect spots using YOLO
        results = self.yolo_model(image_path, verbose=False)
        boxes = results[0].boxes
        
        if len(boxes) == 0:
            return {"error": "No spots detected in the image"}
        
        # Step 2: Calculate Rf values
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        
        # Estimate reference lines (you can improve this with line detection)
        y_origin = int(h * 0.85)
        y_front = int(h * 0.15)
        travel_distance = y_origin - y_front
        
        rf_values = []
        spot_positions = []
        
        for box in boxes:
            # Get center Y position of detected spot
            y_center = float(box.xyxy[0][1] + box.xyxy[0][3]) / 2
            x_center = float(box.xyxy[0][0] + box.xyxy[0][2]) / 2
            
            # Calculate Rf value
            rf = (y_origin - y_center) / travel_distance
            rf = max(0.0, min(1.0, rf))  # Clamp between 0 and 1
            
            rf_values.append(rf)
            spot_positions.append({'x': int(x_center), 'y': int(y_center)})
        
        rf_values = sorted(rf_values)
        
        # Step 3: Extract features for recommendation model
        if len(rf_values) < 2:
            separations = [0]
        else:
            separations = [rf_values[i+1] - rf_values[i] 
                          for i in range(len(rf_values)-1)]
        
        features = [
            len(rf_values),                    # Number of spots
            np.mean(rf_values),                # Average Rf
            np.std(rf_values),                 # Std of Rf
            min(rf_values),                    # Min Rf
            max(rf_values),                    # Max Rf
            max(rf_values) - min(rf_values),   # Rf range
            np.mean(separations),              # Avg separation
            min(separations)                   # Min separation
        ]
        
        # Step 4: Get recommendations
        X = self.scaler.transform([features])
        X_t = torch.FloatTensor(X)
        
        with torch.no_grad():
            sol_pred, col_pred, time_pred = self.rec_model(X_t)
            
            solvent_class = torch.argmax(sol_pred, dim=1).item()
            num_columns = int(round(col_pred.item()))
            est_time = int(round(time_pred.item()))
        
        # Map solvent class to name
        solvent_systems = {
            0: "Non-polar (Hexane/Ethyl Acetate 8:2)",
            1: "Medium polarity (DCM/Methanol 9:1)",
            2: "Polar (Ethyl Acetate/Methanol 7:3)"
        }
        
        # Step 5: Return results
        return {
            "num_spots_detected": len(rf_values),
            "rf_values": [round(rf, 3) for rf in rf_values],
            "spot_positions": spot_positions,
            "separation_quality": {
                "average_separation": round(np.mean(separations), 3),
                "minimum_separation": round(min(separations), 3),
                "rf_range": round(max(rf_values) - min(rf_values), 3)
            },
            "recommendations": {
                "solvent_system": solvent_systems[solvent_class],
                "num_columns_needed": max(1, num_columns),
                "estimated_time_minutes": est_time,
                "estimated_time_hours": round(est_time / 60, 1)
            }
        }
    
    def analyze_and_visualize(self, image_path, output_path='result.jpg'):
        """Analyze and create visualization"""
        
        # Get analysis results
        results = self.analyze_tlc(image_path)
        
        if "error" in results:
            print(f"Error: {results['error']}")
            return results
        
        # Create visualization
        img = cv2.imread(image_path)
        yolo_results = self.yolo_model(image_path, verbose=False)
        annotated = yolo_results[0].plot()
        
        # Add Rf values to image
        for i, (pos, rf) in enumerate(zip(results['spot_positions'], 
                                          results['rf_values'])):
            cv2.putText(annotated, f"Rf:{rf}", 
                       (pos['x'] + 10, pos['y']), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        cv2.imwrite(output_path, annotated)
        print(f"✓ Visualization saved to: {output_path}")
        
        return results


# Example usage
if __name__ == "__main__":
    # Initialize analyzer
    analyzer = TLCAnalyzer(
        yolo_model_path='models/yolo_spot_detector.pt',
        recommendation_model_path='models/recommendation_model.pth'
    )
    
    # Analyze an image
    result = analyzer.analyze_tlc('your_tlc_image.jpg')
    
    # Print results
    print(json.dumps(result, indent=2))
    
    # Create visualization
    analyzer.analyze_and_visualize('your_tlc_image.jpg', 'output_result.jpg')
