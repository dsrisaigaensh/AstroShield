#!/usr/bin/env python3
"""
Model Evaluation Script
Evaluate trained model performance with metrics
"""

import pandas as pd
import os

def evaluate_model():
    """
    Evaluate model using training results
    """
    results_file = "runs/detect/safety_detector/results.csv"
    
    if not os.path.exists(results_file):
        print("❌ No training results found!")
        print("Please run training first: python3 run_train.py")
        return
    
    try:
        # Read the results CSV
        df = pd.read_csv(results_file)
        
        # Get the last epoch results (final metrics)
        final_metrics = df.iloc[-1]
        
        print("📊 Model Evaluation Results:")
        print("=" * 50)
        print(f"Epoch:           {final_metrics['epoch']:.0f}")
        print(f"Training Time:   {final_metrics['time']:.1f} seconds")
        print()
        print("📈 Performance Metrics:")
        print(f"Precision:       {final_metrics['metrics/precision(B)']:.4f}")
        print(f"Recall:          {final_metrics['metrics/recall(B)']:.4f}")
        print(f"mAP@0.5:         {final_metrics['metrics/mAP50(B)']:.4f}")
        print(f"mAP@0.5:0.95:    {final_metrics['metrics/mAP50-95(B)']:.4f}")
        
        # Calculate F1-Score
        precision = final_metrics['metrics/precision(B)']
        recall = final_metrics['metrics/recall(B)']
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"F1-Score:        {f1_score:.4f}")
        
        # Performance interpretation
        print("\n📋 Performance Interpretation:")
        if final_metrics['metrics/mAP50(B)'] > 0.7:
            print("🟢 Excellent performance!")
        elif final_metrics['metrics/mAP50(B)'] > 0.5:
            print("🟡 Good performance")
        elif final_metrics['metrics/mAP50(B)'] > 0.3:
            print("🟠 Moderate performance - consider more training")
        else:
            print("🔴 Low performance - needs more training or data")
            
    except Exception as e:
        print(f"❌ Error reading results: {e}")

if __name__ == "__main__":
    evaluate_model()