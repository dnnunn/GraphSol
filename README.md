# GraphSol
A Protein Solubility Predictor developed by Graph Convolutional Network and Predicted Contact Map

The source code for our paper [Structure-aware protein solubility prediction from sequence through graph convolutional network and predicted contact map](https://jcheminf.biomedcentral.com/articles/10.1186/s13321-021-00488-1)

## 0. Update(2025-07-21) - Tested Installation

**Successfully tested and verified on Google Cloud VM with NVIDIA L4 GPU**

This installation has been tested and confirmed working with the following setup:
- **Environment**: Google Cloud Platform VM
- **GPU**: NVIDIA L4 (with CPU fallback support)
- **Python**: 3.7.9 via conda
- **PyTorch**: 1.6.0
- **Status**: ✅ All tests passing with strong performance metrics

### Key Fixes Applied:
- Fixed model architecture compatibility (GCN_FEATURE_DIM: 91 → 94)
- Fixed feature loading to retain all 94 dimensions
- Added proper .pkl file filtering
- CUDA device compatibility (device 1 → 0)
- Complete data extraction workflow

## 0. Update(2022-03-01)
We have reimplemented the [GraphSol](https://github.com/jcchan23/SAIL/tree/main/Repeat/GraphSol) model by using dgl, which have been optimized in training time and costing memory without losing the accuracy.

## TODO

- [x] Merge the prediction workflow into the original workflow.
- [x] Batch size > 1 in the reimplemention.
- [x] Fix model architecture compatibility issues
- [x] Add comprehensive installation instructions

## 1. Dependencies
The code has been tested under Python 3.7.9, with the following packages installed (along with their dependencies):
- torch==1.6.0
- numpy==1.19.1
- scikit-learn==0.23.2
- pandas==1.1.0
- tqdm==4.48.2

## 2. How to retrain the GraphSol model and test?
If you want to reproduce our result, please refer to the steps below.

### Step 1: Create Conda Environment
```bash
# Create isolated environment for GraphSol
conda create -n GraphSol python=3.7.9
conda activate GraphSol

# Install required packages
pip install torch==1.6.0 numpy==1.19.1 scikit-learn==0.23.2 pandas==1.1.0 tqdm==4.48.2
```

### Step 2: Download Feature Files
Download the required feature files from Google Drive:
- **Google Drive Link**: https://drive.google.com/drive/folders/1ZfeQtLPtRuHeTtA-Iex4Bs2HrrFz25YX?usp=sharing

```bash
# Install gdown for Google Drive downloads
pip install gdown

# Download feature files
gdown --folder https://drive.google.com/drive/folders/1ZfeQtLPtRuHeTtA-Iex4Bs2HrrFz25YX?usp=sharing
```

### Step 3: Extract All Feature Files
```bash
# Extract downloaded files to Data directory
unzip GraphSol/node_features.zip -d Data/
unzip GraphSol/edge_features.zip -d Data/
unzip Data/fasta.zip -d Data/

# Verify extraction
ls -la Data/node_features/ | head -5
ls -la Data/edge_features/ | head -5
ls -la Data/fasta/ | head -5
```

### Step 4: Verify Setup
```bash
# Check that all required files are present
ls -la Data/
# Should show: node_features/, edge_features/, fasta/, eSol_test.csv, eSol_train.csv, etc.

# Check model files
ls -la Model/
# Should show: best_model.pkl and Fold*_best_model.pkl files
```

### Step 3: Run the training code
Run the following python script and it will take about 1 hour to train the model.
```
$ python Train.py
```
A trained model will be saved in the folder `./Model` and validation results in the folder `./Result`

### Step 5: Run the test code
Run the following python script and it will be finished in a few seconds.

**For CPU-only execution (recommended for compatibility):**
```bash
export CUDA_VISIBLE_DEVICES=""
python Test.py
```

**For GPU execution (if CUDA is available):**
```bash
unset CUDA_VISIBLE_DEVICES
python Test.py
```

**Expected Output:**
```
best_model.pkl
100%|████████████████████| 783/783 [00:XX<00:00, XX.XXit/s]
========== Evaluate Test set ==========
Test loss:  0.23065098684155594
Test pearson: (0.6986683018529942, 1.0839972334993266e-115)
Test r2: 0.4832564860545563
Test binary acc:  0.7713920817369093
Test precision: 0.7746031746031746
Test recall:  0.6931818181818182
Test f1:  0.7316341829085456
Test auc:  0.8656797089221683
Test mcc:  0.5360861010583435
```

## 2.1 Troubleshooting

### Common Issues and Solutions:

**1. Model Size Mismatch Error:**
```
RuntimeError: size mismatch for gcn.gc1.weight: copying a param with shape torch.Size([94, 256]) from checkpoint, the shape in current model is torch.Size([91, 256])
```
**Solution:** Ensure `GCN_FEATURE_DIM = 94` in `Test.py` (not 91)

**2. Feature Dimension Mismatch:**
```
RuntimeError: size mismatch, m1: [359 x 91], m2: [94 x 256]
```
**Solution:** Ensure the `load_features()` function returns all 94 dimensions without column slicing

**3. UnpicklingError:**
```
_pickle.UnpicklingError: invalid load key, '\x0a'
```
**Solution:** Add `.pkl` file filtering in the model loading loop to skip text files

**4. CUDA Device Error:**
```
RuntimeError: CUDA error: device-side assert triggered
```
**Solution:** Use CPU-only mode with `export CUDA_VISIBLE_DEVICES=""`

**5. Missing Feature Files:**
```
FileNotFoundError: [Errno 2] No such file or directory: './Data/node_features/...'
```
**Solution:** Ensure all 3 zip files are extracted: `node_features.zip`, `edge_features.zip`, `fasta.zip`

## 3. How to predict protein solubility by the pretrained GraphSol model?

**Note:**

**This is a demo for prediction that contains of 5 protein sequences `aaeX, aas, aat, abgA, abgB` with their preprocessed feature files. You can directly use `$ python predict.py`, and then the result file will be generated in `./Predict/Result/result.csv` with the output format:**

| name | prediction | sequence |
| -------- | -------- | -------- |
| aaeX | 0.3201722800731659 | MSLFPVIVVFGLSFPPIFFELLLSLAIFWLVRRVLVPTGIYDFVWHPALFNTALYC... |
| aas | 0.2957891821861267 | MLFSFFRNLCRVLYRVRVTGDTQALKGERVLITPNHVSFIDGILLGLFLPVRPVFA... |
| ... | ... | ... |

If you want to predict your own protein sequences with using our pretrained models please refer to the steps below.

### Step 1: Prepare your single fasta files
For each protein sequence, you should prepare a corresponding fasta file.

We follow the common fasta file format that starts with `>{protein sequence name}`, then a protein sequence of 80 amino acid letters within one row. This is our demo in `/Data/source/abgB`.

```
>abgB
MQEIYRFIDDAIEADRQRYTDIADQIWDHPETRFEEFWSAEHLASALESAGFTVTRNVGNIPNAFIASFGQGKPVIALL
GEYDALAGLSQQAGCAQPTSVTPGENGHGCGHNLLGTAAFAAAIAVKKWLEQYGQGGTVRFYGCPGEEGGSGKTFMVRE
GVFDDVDAALTWHPEAFAGMFNTRTLANIQASWRFKGIAAHAANSPHLGRSALDAVTLMTTGTNFLNEHIIEKARVHYA
ITNSGGISPNVVQAQAEVLYLIRAPEMTDVQHIYDRVAKIAEGAALMTETTVECRFDKACSSYLPNRTLENAMYQALSH
FGTPEWNSEELAFAKQIQATLTSNDRQNSLNNIAATGGENGKVFALRHRETVLANEVAPYAATDNVLAASTDVGDVSWK
LPVAQCFSPCFAVGTPLHTWQLVSQGRTSIAHKGMLLAAKTMAATTVNLFLDSGLLQECQQEHQQVTDTQPYHCPIPKN
VTPSPLK
```

**Note:**

(1) Please name your protein sequence uniquely and as short as possible, since the protein sequence name will be used as the file name in the step 3, such as `abgB.pssm`, `abgB.spd33`.

(2) Please name your fasta file **without** using any suffix, such as `abgB` instead of `abgB.fasta` or `abgB.fa`, otherwise the feature generation software in the step 3 will name the feature file with the format of `abgB.fasta.pssm` or `abgB.fa.pssm`, leading to unexpected error.

### Step 2: Prepare your total fasta file
We follow the common fasta file format that starts with `>{protein sequence name}`, hen a protein sequence of 80 amino acid letters within one row. This is part of our demo in `./Data/upload/input.fasta`.

```
>aat
MRLVQLSRHSIAFPSPEGALREPNGLLALGGDLSPARLLMAYQRGIFPWFSPGDPILWWSPDPRAVLWPESLHISRSMK
RFHKRSPYRVTMNYAFGQVIEGCASDREEGTWITRGVVEAYHRLHELGHAHSIEVWREDELVGGMYGVAQGTLFCGESM
FSRMENASKTALLVFCEEFIGHGGKLIDCQVLNDHTASLGACEIPRRDYLNYLNQMRLGRLPNNFWVPRCLFSPQE
>abgA
MESLNQFVNSLAPKLSHWRRDFHHYAESGWVEFRTATLVAEELHQLGYSLALGREVVNESSRMGLPDEFTLQREFERAR
QQGALAQWIAAFEGGFTGIVATLDTGRPGPVMAFRVDMDALDLSEEQDVSHRPYRDGFASCNAGMMHACGHDGHTAIGL
GLAHTLKQFESGLHGVIKLIFQPAEEGTRGARAMVDAGVVDDVDYFTAVHIGTGVPAGTVVCGSDNFMATTKFDAHFTG
TAAHAGAKPEDGHNALLAAAQATLALHAIAPHSEGASRVNVGVMQAGSGRNVVPASALLKVETRGASDVINQYVFDRAQ
QAIQGAATMYGVGVETRLMGAATASSPSPQWVAWLQSQAAQVAGVNQAIERVEAPAGSEDATLMMARVQQHQGQASYVV
FGTQLAAGHHNEKFDFDEQVLAIAVETLARTALNFPWTRGI
```

### Step 3: Prepare 5 node feature files and 1 edge feature file
**Note:**

(1) We don't integrate the feature generation software in our repository, please use the recommend software(see the table below) to generate the feature files !!!

(2) We have deployed all feature generation softwares in our servers to calculate the features in bulk, the link below is utilized to map the sequence files to feature files as an example.

(3) In the software SPOT-Contact, it needs a sequence file with suffix `.fasta`, thus you should rename the original fasta file `abgB` to `abgB.fasta` after generating other features.

(4) **THIS STEP WILL COST MOST OF THE TIME !!!!!** (The sequence with more amino acids will cost longer time, so we recommend to use the protein sequence less than 700 amino acids.)

| Software | Version | Input | Output |
| -------- | -------- | -------- | --------|
| [PSI-BLAST](https://blast.ncbi.nlm.nih.gov/Blast.cgi?PAGE_TYPE=BlastSearch&PROGRAM=blastp&BLAST_PROGRAMS=psiBlast) | v2.7.1 | abgB | abgB.bla, abgB.pssm |
| [HH-Suite3](https://github.com/soedinglab/hh-suite) | v3.0.3 | abgB | abgB.hhr, abgB.hhm, abgB.a3m |
| [SPIDER3](https://sparks-lab.org/server/spider3/) | v1.0 | abgB, abgB.pssm, abgB.hhm | abgB.spd33 |
| [DCA](http://dca.rice.edu/portal/dca/) | v1.0 | abgB.a3m | abgB.di |
| [CCMPred](https://github.com/soedinglab/CCMpred) | v1.0 | abgB.a3m | abgB.mat |
| [SPOT-Contact](https://sparks-lab.org/server/spot-contact/) | v1.0 | abgB.fasta, abgB.pssm, abgB.hhm, abgB.di, abgB.mat | abgB.spotcon |

Then put all the generated files into the folder `./Data/source/`(We have provided a list of files as an example). Other precautions when using the feature generation software please refer to the corresponding software document.

### Step 4: Run the predict code
```
$ python predict.py
```
All the prediction result will be stored as in `./Result/result.csv`.

## 4. The web server of the GraphSol model
Our platform are highly recommended to be academicly used only (e.g. for limited protein sequences).

[https://biomed.nscc-gz.cn/apps/GraphSol](https://biomed.nscc-gz.cn/apps/GraphSol)


## 5. How to train the GraphSol model with your own data? 
If you want to train a model with your own data:

(1) Please refer to the feature generation steps to preprocess 6 feature files. 

(2) Use `get1D_features.py` and `get2D_features.py` to generate two matrices, and then move them to the folders `./Data/node_features` and `./Data/edge_features`, respectively. 

(3) Make a general csv file with the format like `./Data/eSol_train.csv` or `./Data/eSol_test.csv`.

(4) Run `$ python Train.py`, and optionly tune the hypermeters in the same file.

## 6. Citations
Please cite our paper if you want to use our code in your work.
```bibtex
@article{chen2021structure,
  title={Structure-aware protein solubility prediction from sequence through graph convolutional network and predicted contact map},
  author={Chen, Jianwen and Zheng, Shuangjia and Zhao, Huiying and Yang, Yuedong},
  journal={Journal of cheminformatics},
  volume={13},
  number={1},
  pages={1--10},
  year={2021},
  publisher={Springer}
}
