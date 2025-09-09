# An Integrated Framework for GNSS Trajectory Field–Road Classification with DBSCAN-Guided Augmentation and Spatiotemporal Learning
This study addresses the challenge of field–road classifica-tion in precision agriculture using GNSS trajectory data. Accurate identification of operational (field) and travel (road) points is critical for smart agriculture, land-use monitoring, and field management.
#  Journal Extensions
Compared with our ICIC 2025 conference paper “ VE-ResBiLSTM: A Deep Spatiotemporal Model for Field–Road Classification with DBSCAN-Based Data Augmentation”, the journal version achieves substantial extensions and systematic improvements in theoretical depth, methodological design, experimental analysis, and reproducibility:
1. Updated Framework/Pipeline: The conference version employed smoothing in preprocessing, but subsequent analysis showed that smoothing could introduce artificial points not present in the raw GNSS data. In this extended version, we revised and updated the entire framework/pipeline, removing smoothing to ensure data fidelity and methodological rigor. This correction represents both a key motivation and a fundamental improvement of the work.
2. Theoretical Enhancement of DBSCAN-Guided Augmentation: We formalize the interpolation-based augmentation with parameter sensitivity analysis, K-distance calibration, and mathematical derivations, establishing a theoretical basis for trajectory data augmentation.
3. Model Transparency: Detailed architectural explanations and diagrams are added for the VAE and Res-BiLSTM modules, clarifying mechanisms and strengthening theoretical grounding.
4. Comprehensive Ablation Studies: Extensive ablation experiments systematically validate the contributions of DBSCAN augmentation, VAE, and ResBiLSTM, with confusion matrices to illustrate classification effectiveness.
5. Data-Level Expansion & Evaluation Standard: Two additional agricultural machinery datasets (Harvester, Tractor) are incorporated to test generalization across regions and machine types. Moreover, we propose a trajectory augmentation evaluation standard, introducing Fr´echet, Hausdorff, and LCSS similarity metrics to assess the plausibility and consistency of augmented trajectories. This provides a methodological contribution beyond classification accuracy.
6. Comparative Benchmarks: Mainstream oversampling methods (SMOTE, ADASYN) are included as baselines, further reinforcing the distinct advantages of the DBSCAN-guided approach.

# Datasets
Wheat,Paddy,Corn,Tractor: https://github.com/Agribigdata/dataset_code.
Harvester: https://github.com/AgriMachineryBigData/Field-road_mode_mining.

# Version Information
1.OS: Ubuntu 20.04 
2.CUDA: 12.4 
3.PyTorch: 2.5.1
4.Python: 3.10.16
GPU: NVIDIA A100

