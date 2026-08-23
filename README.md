# Revisiting Subject-Independent Evaluation for Smartphone-Based Fall and Activity Recognition

This repository contains the official code for the paper:  
**Revisiting Subject-Independent Evaluation for Smartphone-Based Fall and Activity Recognition: A Lightweight Benchmark on MobiAct**  
*Zainab Raza Malik and Muhammad Zeeshan Abbas*

## Overview

Human activity recognition (HAR) models evaluated using random cross-validation (CV) can produce optimistic performance estimates due to subject-mixed data partitions. This project systematically quantifies the CV-to-leave-one-subject-out (LOSO) generalization gap on the MobiAct dataset across four task formulations:
- Binary (Fall vs. ADL)
- Fall-type classification (4 classes)
- ADL-only classification (11 classes)
- Full activity recognition (15 classes)

To reduce this cross-subject performance loss, we introduce **Feature-Invariance-Conditioned (FIC) pooling** paired with **Sharpness-Aware Minimization (SAM)**. The combination provides physically guided temporal representation learning and optimization-level regularization within a lightweight model (81.5k parameters), effectively reducing the generalization gap.

## Dataset Availability

Due to data usage agreements, the MobiAct dataset cannot be redistributed in this repository. You must acquire it directly from the original authors.

1. Request access to MobiAct v2.0 from the Hellenic Mediterranean University (Biomedical Informatics and eHealth Laboratory): [MobiAct Dataset](https://bmi.hmu.gr/the-mobifall-and-mobiact-datasets-2/)
2. Once downloaded, extract the dataset and locate the `Annotated Data` folder.
3. Place the `Annotated Data` folder inside the `data/` directory of this repository so the structure looks like:
   ```
   data/
   └── Annotated Data/
       ├── BSC/
       ├── CHU/
       ├── CSI/
       ...
   ```

## Dependencies

- Python 3.8+
- PyTorch (>= 1.10)
- NumPy
- Pandas
- scikit-learn

You can install the required packages using pip:
```bash
pip install torch numpy pandas scikit-learn
```

## Running the Evaluation

The entire evaluation pipeline (preprocessing, model training across all four modes, and CV/LOSO evaluation) is managed by `src/main.py`.

```bash
python src/main.py
```

The script will:
1. Load and resample raw sensor data (200Hz to 50Hz).
2. Segment signals into 1.0s windows with 50% overlap.
3. Train and evaluate four configurations (`ERM`, `SAM`, `FIC`, `SAM+FIC`) on both CV and LOSO protocols.
4. Output results iteratively to `results/01_primary_eval/`.

**Resume Support:** If the run is interrupted, the script will skip completed configurations and resume from the last saved checkpoint automatically.

## Code Structure

- `src/config.py`: Global hyperparameter configuration, modes, and task definitions.
- `src/data/loader.py`: Raw signal resampling, windowing, and dataset construction.
- `src/data/features.py`: Handcrafted physical features for FIC (jerk, tilt, spectral, etc.).
- `src/models/cnn.py`: `LightCNN` (ERM/SAM backbone) and `FICNet` (FIC pooling backbone).
- `src/models/sam.py`: Sharpness-Aware Minimization optimizer wrapper.
- `src/training/train.py`: Training loop encompassing CE and FIC consistency losses.
- `src/training/evaluate.py`: Stratified CV and pooled LOSO evaluation protocols.
- `src/main.py`: Main entry point orchestrating tasks, modes, protocols, and checkpointing.

## Citation

If you use this codebase or find our work helpful, please cite our paper:

```bibtex
@article{malik2026revisiting,
  title={Revisiting Subject-Independent Evaluation for Smartphone-Based Fall and Activity Recognition: A Lightweight Benchmark on MobiAct},
  author={Malik, Zainab Raza and Abbas, Muhammad Zeeshan},
  year={2026}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
