#!/usr/bin/env python3
"""
GraphSol Predictor Wrapper (Corrected)
Standardized interface for GraphSol solubility prediction.
Uses the actual GraphSol Predict/predict.py workflow.
"""

import argparse
import pandas as pd
import sys
import os
import subprocess
import tempfile
import shutil
from Bio import SeqIO

WRAPPER_PREDICTOR_NAME = "GraphSol"

def run_graphsol_prediction(fasta_path, output_path):
    """
    Run GraphSol prediction using the existing Predict/predict.py pipeline.
    """
    
    # GraphSol prediction directory
    graphsol_predict_dir = "/home/david_nunn/GraphSol/Predict"
    
    # Ensure the upload directory exists
    upload_dir = os.path.join(graphsol_predict_dir, "Data/upload")
    os.makedirs(upload_dir, exist_ok=True)
    
    # Copy input FASTA to GraphSol's expected location
    input_fasta = os.path.join(upload_dir, "input.fasta")
    shutil.copy2(fasta_path, input_fasta)
    
    # Ensure Result directory exists
    result_dir = os.path.join(graphsol_predict_dir, "Result")
    os.makedirs(result_dir, exist_ok=True)
    
    # Run GraphSol prediction from the Predict directory
    cmd = ["python", "predict.py"]
    result = subprocess.run(
        cmd, 
        cwd=graphsol_predict_dir,
        capture_output=True, 
        text=True,
        timeout=3600  # 1 hour timeout
    )
    
    if result.returncode != 0:
        print(f"GraphSol stdout: {result.stdout}")
        print(f"GraphSol stderr: {result.stderr}")
        raise RuntimeError(f"GraphSol prediction failed with return code {result.returncode}")
    
    # GraphSol outputs to Result/result.csv
    graphsol_output = os.path.join(result_dir, "result.csv")
    
    if not os.path.exists(graphsol_output):
        raise FileNotFoundError(f"GraphSol output not found: {graphsol_output}")
    
    # Read GraphSol output and standardize format
    df = pd.read_csv(graphsol_output)
    
    # Read original sequences for mapping
    sequences = {}
    for record in SeqIO.parse(fasta_path, "fasta"):
        sequences[record.id] = str(record.seq)
    
    # Create standardized output
    output_rows = []
    for _, row in df.iterrows():
        seq_id = row['name']
        solubility_score = float(row['prediction'])  # GraphSol outputs continuous scores
        
        # Convert to probabilities (GraphSol score should be 0-1 range)
        prob_soluble = solubility_score
        prob_insoluble = 1.0 - solubility_score
        
        output_rows.append({
            'Accession': seq_id,
            'Sequence': sequences.get(seq_id, ''),
            'Predictor': WRAPPER_PREDICTOR_NAME,
            'SolubilityScore': solubility_score,
            'Probability_Soluble': prob_soluble,
            'Probability_Insoluble': prob_insoluble
        })
    
    # Write standardized output
    output_df = pd.DataFrame(output_rows)
    output_df.to_csv(output_path, index=False)
    
    print(f"GraphSol prediction completed. Results saved to {output_path}")
    print(f"Processed {len(output_rows)} sequences")

def main():
    parser = argparse.ArgumentParser(description='GraphSol Solubility Predictor Wrapper')
    parser.add_argument('--fasta', required=True, help='Input FASTA file')
    parser.add_argument('--out', required=True, help='Output CSV file')
    
    args = parser.parse_args()
    
    try:
        run_graphsol_prediction(args.fasta, args.out)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
