#!/usr/bin/env python3
"""
GraphSol Predictor Wrapper (Fixed)
Standardized interface for GraphSol solubility prediction.
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
    Run GraphSol prediction using the existing standalone pipeline.
    GraphSol requires feature preprocessing but has a working predict.py.
    """
    
    # GraphSol prediction directory
    graphsol_predict_dir = "/home/david_nunn/GraphSol/Predict"
    
    # Copy input FASTA to GraphSol's expected location
    input_fasta = os.path.join(graphsol_predict_dir, "Data/upload/input.fasta")
    os.makedirs(os.path.dirname(input_fasta), exist_ok=True)
    shutil.copy2(fasta_path, input_fasta)
    
    # Run GraphSol prediction
    cmd = ["python", "predict.py"]
    result = subprocess.run(
        cmd, 
        cwd=graphsol_predict_dir,
        capture_output=True, 
        text=True,
        timeout=3600  # 1 hour timeout
    )
    
    if result.returncode != 0:
        raise RuntimeError(f"GraphSol prediction failed: {result.stderr}")
    
    # GraphSol outputs to Result/result.csv
    graphsol_output = os.path.join(graphsol_predict_dir, "Result/result.csv")
    
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
        
        # Convert to probabilities (assuming GraphSol score is already 0-1 range)
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
