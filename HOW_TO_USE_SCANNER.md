# 🔬 How to Use the Interactive Mutation Scanner

The Mutation Scanner is a powerful new tool in the Protein Design Hub that allows you to rapidly screen single-point mutations using ESMFold. It provides immediate structural feedback and stability metrics (pLDDT) to help you choose the best variants.

## 🚀 Getting Started

You can access the Mutation Scanner in two ways:
1. **Direct Access**: Click on `Mutation Scanner` in the sidebar navigation.
2. **From Designer**: In the `Design` page, select a single residue and click "Run Mutation Scanner on Pos X".

## 📋 Workflow

### 1. Input Sequence & Base Prediction
- Paste your sequence or load an example.
- Click **"Predict Base Structure"** to run ESMFold on your wild-type sequence.
- **Auto-Prediction**: If you come from the Design page, the sequence is pre-loaded.

### 2. Select a Position
- The interactive grid shows your sequence colored by pLDDT confidence (blue=high, orange=low).
- Click any residue to select it for scanning.
- **Tip**: Target low-confidence regions (orange/yellow) to find stabilizing mutations.

### 3. Run Saturation Mutagenesis
- Click **"Scan All Mutations"** to automatically:
  - Generate all 19 possible amino acid substitutions.
  - Predict 3D structures for every mutant using ESMFold.
  - Calculate pLDDT changes (ΔpLDDT) relative to the wild-type.
  - Rank mutations by improvement score.

### 4. Analyze Results

#### 🏆 Recommendations
- See the top recommended mutations that improve stability.
- **Green cards**: Beneficial mutations (increase pLDDT).
- **Red cards**: Destabilizing mutations.

#### 🗺️ Heatmap
- Visual overview of how every amino acid affects the position.
- Hover over bars to see detailed scores.

#### 📊 Detailed Metrics
- Sortable table with:
  - **Mean pLDDT**: Global stability score.
  - **Local pLDDT**: Stability at the mutation site.
  - **RMSD**: Structural deviation from wild-type (check for backbone shifts).

#### 🔬 Structure Comparison
- View Wild-Type and Mutant structures side-by-side.
- Rotate and zoom to inspect side-chain packing.

## 📤 Export & Next Steps
- **Download Variants**: Get a FASTA file with the top recommended sequences.
- **Run ColabFold**: Use the "Go to Predict Page" button to take your best variants and run them through high-accuracy predictors like ColabFold, Chai-1, or Boltz-2 for final validation.

## 💡 Key Metrics Explained

| Metric | Description | Good Value |
|--------|-------------|------------|
| **mean pLDDT** | Average predicted confidence of the whole structure (0-100). | > 0 change |
| **local pLDDT** | Predicted confidence specifically at the mutated position. | > 0 change |
| **RMSD** | Root Mean Square Deviation of atomic positions between WT and Mutant. | Low (< 2.0Å) usually means backbone is preserved. |
| **Improvement Score** | A weighted combination of global and local pLDDT improvements. | The higher the better. |

---
*Note: The scanner uses the ESMFold API for speed only for sequences < 400 residues. For longer sequences, ensure you have a local GPU environment set up.*
