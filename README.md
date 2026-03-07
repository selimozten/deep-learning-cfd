# akis

Neural network architectures for accelerating computational fluid dynamics simulations.

"Akis" means "flow" in Turkish.

## What It Does

akis applies deep learning to CFD problems: solving Navier-Stokes equations with PINNs, predicting flow fields with CNNs, analyzing unstructured meshes with GNNs, and generating meshes with GANs.

## Approaches

### Physics-Informed Neural Networks (PINNs)
Solve Navier-Stokes equations by embedding physics constraints directly into the loss function. No mesh required — the network learns the continuous solution field.

### CNN Flow Prediction
Predict steady-state flow fields from geometry inputs. Train on simulation data, infer in milliseconds instead of hours.

### GNN Mesh Analysis
Operate on unstructured meshes directly using graph neural networks. Predict where mesh refinement is needed without running the full simulation.

### GAN Mesh Generation
Generate high-quality computational meshes from geometry descriptions. Trained on meshes from existing simulations.

## Project Structure

```
akis/
  data/
    cfd_simulations/
    meshes/
    processed/
  notebooks/
    1_pinn_navier_stokes.ipynb
    2_cnn_flow_prediction.ipynb
    3_gnn_mesh_analysis.ipynb
    4_gan_mesh_generation.ipynb
  src/
    models/
      pinn/
      cnn/
      gnn/
      gan/
    data_processing/
    visualization/
    utils/
  tests/
  requirements.txt
```

## Setup

```bash
git clone https://github.com/selimozten/akis.git
cd akis
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

Requires a CUDA-capable GPU for training.

## Tech Stack

- PyTorch for model training
- NumPy / SciPy for numerical computations
- OpenFOAM for validation against traditional CFD
- NetworkX for graph representations

## References

1. Raissi et al. (2019) — Physics-informed neural networks for solving PDEs
2. Guo et al. (2016) — CNNs for steady flow approximation
3. Pfaff et al. (2021) — Learning mesh-based simulation with graph networks
4. Chen & Jaiman (2022) — MeshingNet for deep learning mesh generation

## License

Apache 2.0
