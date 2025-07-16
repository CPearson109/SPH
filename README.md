Real-Time Multi-Phase Fluid Simulation
This project demonstrates a GPU-accelerated, Smoothed Particle Hydrodynamics (SPH)-based simulation capable of rendering real-time interactions between multi-phase fluids (e.g., oil and water). Developed using Unity's compute shader pipeline, this project showcases optimisations in spatial partitioning, fluid interface dynamics, and visual rendering fidelity.

Table of Contents
Project Overview

Features

Getting Started

Folder Structure

Dependencies

Usage

Key Scripts

Known Issues

Author

Project Overview
This system implements real-time SPH simulation to model multi-phase fluid dynamics, focusing on performance and visual accuracy. It supports fluid interaction, boundary constraints, surface tension effects, and real-time visualisation.

Features
Real-time GPU-accelerated SPH solver

Support for multiple fluid types with different physical properties

Surface tension modelling with dynamic thresholding

Efficient neighbour search via spatial grids

XSPH velocity smoothing

Procedural particle spawning from multiple sources

Visual rendering using Unity shader pipeline

Getting Started
Open the project in Unity 2022.3 LTS or newer.

Attach SPH.cs to an empty GameObject in your scene.

Create and assign SpawnBox GameObjects to initialise fluid regions.

Assign a boundary cube and optional colliders for obstacles.

Run the scene.

Folder Structure
markdown
Copy
Edit
/Assets
  /Scripts
    - SPH.cs
    - SpawnBox.cs
  /Shaders
    - SPHCompute.compute
    - ParticleRender.shader
Dependencies
Unity 2022.3 LTS

Compute shader support (DX11+ or Metal)

Compatible GPU (NVIDIA GTX 10xx or newer recommended)

Usage
Adjust fluid parameters (e.g., viscosity, mass) in the SpawnBox script.

Set simulation bounds via the boundary cube or fallback values.

Enable debugging visuals using Gizmos in the Unity editor.

Key Scripts
SPH.cs: Core simulation manager, handling particle buffers, grid generation, and compute dispatch.

SpawnBox.cs: Defines spawn volumes and particle properties per fluid type.

SPHCompute.compute: Compute shader handling SPH kernel calculations and particle updates.

ParticleRender.shader: Shader for rendering particles as fluid points.

Known Issues
Simulation performance may degrade with more than ~200,000 particles on mid-tier GPUs.

Certain surface tension configurations can lead to instability if parameters are not tuned.

Author
Cameron Pearson
40530119 – BSc (Hons) Computing
Edinburgh Napier University, 2025

