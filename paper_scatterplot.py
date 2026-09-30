"""
Distance with Distance Comparison Report (Matrix Version)
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error
import dcor # For Distance Correlation

# --- 1. DATA LOADING AND CONFIGURATION ---

# --- 1.1 Configuration ---
# Define the maximum number of unique distance pairs to use for speed/RAM
MAX_SAMPLED_PAIRS = 500000 

# --- 1.2 File Paths (YOUR INPUT) ---
PATH_D_ORIGINAL = 'outputs/WLASL_npy_dataset_100_split/L2_r/D_val_val.npy' # Your D_L2
PATH_D_NEW = 'outputs/WLASL_npy_dataset_100_split/Quat_r/D_val_val.npy'     # Your D_Quat


# --- 2. DATA LOADING AND PREPARATION ---

def load_and_prepare_matrices(path_d_orig, path_d_new):
    """Loads, checks for consistency, flattens, cleans, and subsamples matrices."""
    
    print(f"Loading D_Original (L2): {path_d_orig}")
    D_original = np.load(path_d_orig)
    print(f"Loading D_New (Quat): {path_d_new}")
    D_new = np.load(path_d_new)
    
    if D_original.shape != D_new.shape:
        raise ValueError(
            f"CRITICAL ERROR: Matrices must have the same shape. "
            f"D_Original: {D_original.shape} | D_New: {D_new.shape}"
        )
    
    N = D_original.shape[0]
    
    # 1. Extract the unique pairs (upper triangle, excluding diagonal)
    triu_indices = np.triu_indices(N, k=1)
    D_orig_flat = D_original[triu_indices]
    D_new_flat = D_new[triu_indices]
    
    total_unique_pairs = len(D_orig_flat)
    
    # 2. Create the mask for Inf/NaN (Cleaning)
    finite_mask = np.isfinite(D_orig_flat) & np.isfinite(D_new_flat)
    
    D_orig_clean = D_orig_flat[finite_mask]
    D_new_clean = D_new_flat[finite_mask]
    
    pairs_retained = len(D_orig_clean)
    pairs_dropped = total_unique_pairs - pairs_retained
    
    # 3. Subsampling for Speed
    D_orig_final = D_orig_clean
    D_new_final = D_new_clean
    
    if pairs_retained > MAX_SAMPLED_PAIRS:
        print(f"Subsampling from {pairs_retained} to {MAX_SAMPLED_PAIRS} pairs for speed.")
        np.random.seed(42)
        sample_indices = np.random.choice(pairs_retained, MAX_SAMPLED_PAIRS, replace=False)
        
        D_orig_final = D_orig_clean[sample_indices]
        D_new_final = D_new_clean[sample_indices]
    
    # Convert to float64 for stable/fast dcor calculation
    D_orig_final = D_orig_final.astype(np.float64) 
    D_new_final = D_new_final.astype(np.float64) 
    
    pairs_used = len(D_orig_final)
    
    if pairs_used < 2:
        raise RuntimeError("Not enough finite data points for comparison after cleaning.")

    return D_orig_final, D_new_final, total_unique_pairs, pairs_dropped, pairs_used


# --- 3. CORE REPORTING FUNCTION ---

def generate_custom_comparison_report(D_orig_final, D_new_final, total_unique_pairs, pairs_dropped, pairs_used):
    
    # --- A. Correlation Metrics (RANK and DEPENDENCY) ---
    print("\n" + "="*70)
    print("      CUSTOM DISTANCE MATRIX DIRECT COMPARISON REPORT")
    print("="*70)
    print(f"Matrix Shape: {D_orig_final.shape[0]}x{D_orig_final.shape[0]}")
    print(f"Total Unique Pairs: {total_unique_pairs}")
    print(f"Dropped (Inf/NaN) Pairs:        {pairs_dropped}")
    print(f"Pairs Used for Metrics/Plots:   {pairs_used}")
    
    print("\n[A] Structural and Rank Agreement Metrics:")
    print("-" * 45)
    
    # 1. Spearman's Rho (Measures rank consistency—Crucial for distance comparison)
    spearman_rho, spearman_p_value = spearmanr(D_orig_final, D_new_final)
    print(f"  > Spearman's Rho (Rank Correlation): {spearman_rho:.4f} (Goal: Close to +1)")
    print(f"    (P-value: {spearman_p_value:.2e})")
    
    # 2. Distance Correlation (Measures non-linear dependency)
    try:
        dist_corr = dcor.distance_correlation(D_orig_final, D_new_final)
        print(f"  > Distance Correlation (dCor):     {dist_corr:.4f} (Goal: Close to 1)")
    except Exception as e:
        print(f"  > Distance Correlation (dCor): Calculation failed. ({e})")
        
    # --- B. Numerical Error Metrics ---
    print("\n[B] Numerical Difference Metrics (Meaningful only if scales are similar):")
    print("-" * 45)
    
    # 1. Root Mean Square Error (RMSE)
    rmse = np.sqrt(mean_squared_error(D_orig_final, D_new_final))
    print(f"  > Root Mean Square Error (RMSE):   {rmse:.4f} (Goal: Close to 0)")
    
    # 2. Average Relative Absolute Difference (ARAD)
    arad = np.mean(2 * np.abs(D_new_final - D_orig_final) / (D_new_final + D_orig_final + 1e-6))
    print(f"  > Avg. Relative Abs. Difference:   {arad:.4f} (Goal: Close to 0)")

    # --- C. Visualizations ---
    
    # plt.style.use('ggplot')
    # plt.figure(figsize=(15, 6))

    # 1. Scatter Plot: D_Original vs D_New 
    # plt.subplot(1, 2, 1)
    # plt.scatter(D_orig_final, D_new_final, alpha=0.3, s=5, c=D_orig_final, cmap='viridis')
    # plt.colorbar(label='Original Distance ($D_{L2}$) Magnitude')
    # plt.title(f"Rank Consistency Between Custom Metrics\n(Spearman $\\rho$: {spearman_rho:.2f})")
    # plt.xlabel(f"Original Distance (L2)")
    # plt.ylabel(f"New Distance (D_Quat)")
    # plt.grid(True, linestyle='--', alpha=0.6)


    plt.style.use("ggplot")
    plt.figure(figsize=(15, 6))

    sc = plt.scatter(
        D_orig_final,
        D_new_final,
        s=8,
        alpha=0.4,
        color="tab:blue",    # clean, standard blue
        edgecolors="none",
    )


    plt.xlabel(
        r"L2",
        fontsize=28,
        labelpad=10,
    )
    plt.ylabel(
        r"Quat",
        fontsize=28,
        labelpad=10,
    )


    plt.xticks(fontsize=22)
    plt.yticks(fontsize=22)

    plt.xlim(0, 1.5)     # cap x-axis
    # plt.ylim(0, 1.5)   # optional symmetry

    plt.grid(True, linestyle="--", alpha=0.6)

      
    plt.tight_layout()
    plt.savefig('test.png', dpi=300, bbox_inches="tight")


    print("\n" + "="*70)
    print(" REPORT END")
    print("="*70)


# --- 4. EXECUTION ---
if __name__ == "__main__":
    D_orig_final, D_new_final, total_unique_pairs, pairs_dropped, pairs_used = load_and_prepare_matrices(
        PATH_D_ORIGINAL, PATH_D_NEW
    )
    generate_custom_comparison_report(D_orig_final, D_new_final, total_unique_pairs, pairs_dropped, pairs_used)