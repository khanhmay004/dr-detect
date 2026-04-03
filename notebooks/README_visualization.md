# Results Visualization Notebook

## Overview
**Notebook**: esults_visualization_report.ipynb

This notebook provides a comprehensive visualization of all training and evaluation results from the DR-Detect project.

## What's Included

### 1. **Training Results**
- Training curves (loss, accuracy, kappa, AUC)
- Best epoch metrics
- Per-class performance
- Hyperparameters summary

### 2. **Model Comparison**
- Baseline vs CBAM vs Retrained Baseline
- Internal validation comparison
- Bar charts and tables

### 3. **External Evaluation**
- APTOS test split results
- Messidor-2 external test results
- Per-class breakdowns with heatmaps

### 4. **Domain Shift Analysis**
- Performance across datasets
- Degradation metrics
- Visualization of accuracy/kappa/AUC drops

### 5. **Calibration Analysis**
- Reliability diagrams
- ECE and Brier scores
- Temperature scaling results

### 6. **Uncertainty Analysis**
- Entropy histograms
- Confidence vs entropy scatter plots
- Referral curves (rejection performance)
- Uncertainty CSV statistics

### 7. **Grad-CAM Visualization**
- Sample attention maps
- Summary tables
- Model interpretability

### 8. **Summary Tables**
- Complete results across all models and datasets
- File inventory
- Key findings and conclusions

## How to Use

1. **Start Jupyter**:
   \\\ash
   cd notebooks
   jupyter notebook results_visualization_report.ipynb
   \\\

2. **Run All Cells**:
   - Click **Cell → Run All** to generate the complete report
   - Or run cells individually to explore specific sections

3. **Navigate**:
   - Use the Table of Contents at the top
   - Each section has hyperlinks for easy navigation

## Requirements

The notebook uses standard Python data science libraries:
- pandas
- numpy
- matplotlib
- seaborn
- pathlib
- IPython (for display)

These should already be installed in your environment.

## Output Structure

The notebook reads from:
\\\
outputs/
├── results/       # JSON metrics and CSV uncertainty files
├── logs/          # Training history JSON files
├── figures/       # PNG visualizations
│   └── gradcam/   # Grad-CAM attention maps
└── checkpoints/   # Model weights
\\\

## Key Features

✅ **Automatic Discovery**: Finds all result files automatically  
✅ **Interactive**: Explore data with pandas DataFrames  
✅ **Visual**: High-quality matplotlib/seaborn plots  
✅ **Comprehensive**: Covers training, evaluation, calibration, uncertainty  
✅ **Export-Ready**: All plots can be saved for papers/presentations  

## Tips

- **Slow to load?** Comment out Grad-CAM section if you have many images
- **Custom analysis?** Add your own cells to explore specific aspects
- **Export plots?** Use \plt.savefig('filename.png', dpi=300, bbox_inches='tight')\

## Contact

For issues or questions, refer to the main project documentation.
