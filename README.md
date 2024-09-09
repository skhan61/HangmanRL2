# Hangman Game Solver Using Meta Reinforcement Learning
[![DOI](https://zenodo.org/badge/854590830.svg)](https://zenodo.org/doi/10.5281/zenodo.13737841)

This repository documents a novel approach to solving the Hangman game problem, using advanced meta reinforcement learning techniques, specifically an implementation of the RL² (Reinforcement Learning squared) algorithm, tailored for challenges set by Trexquant.

## Project Overview

The Hangman game solver is developed as part of a challenge to demonstrate the potential of meta reinforcement learning in cognitive and predictive tasks. By employing the RL² algorithm, the model is designed to adapt and generalize effectively to unseen words, enhancing its guessing accuracy under constraints typical of the Hangman game.

## Final Results

The model has successfully achieved an accuracy of **45%** on a challenging set of 1,000 unseen words, which underlines the robustness and efficiency of the meta RL approach in handling complex decision-making scenarios.

## Repository Contents

- `hangman_notebook.ipynb`: An interactive Jupyter notebook for demonstrating and testing the Hangman game solver.
- `Hangman_report.pdf`: A comprehensive report detailing the methodology, implementation specifics, and the results of employing meta RL in this context.
- `rl_squared/`: Directory containing the necessary Python scripts for training and evaluating the RL² model.

## How to Cite

If you use this project in your research or wish to refer to the baseline results, please use the following BibTeX entry:

```bibtex
@misc{hangman_rl_meta_2024,
  author       = {Sayem Khan},
  title        = {{Learning to Learn Hangman}},
  month        = sep,
  year         = 2024,
  doi          = {10.5281/zenodo.13737841},
  version      = {v1.0},
  publisher    = {Zenodo},
  url          = {https://doi.org/10.5281/zenodo.13737841}
}