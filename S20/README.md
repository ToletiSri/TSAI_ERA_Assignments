# Session 20 — Stable Diffusion, Textual Inversion, and Guidance

This session explores the internals of a Stable Diffusion pipeline and modifies generation with learned concepts and a custom guidance loss.

## Core concepts

- **Latent diffusion:** text-conditioned denoising is performed in a compact VAE latent space rather than directly in pixel space.
- **Pipeline components:** the notebook separates the CLIP text encoder, U-Net noise predictor, scheduler, and VAE decoder to expose each inference step.
- **Classifier-free guidance:** conditional and unconditional noise estimates are combined to strengthen prompt adherence.
- **Textual inversion:** learned token embeddings inject community-created visual styles without retraining the diffusion model.
- **Reproducible sampling:** fixed random seeds make style and guidance comparisons meaningful.
- **Custom loss guidance:** a differentiable saturation objective nudges intermediate latents toward less saturated outputs during denoising.

## Experiment

`S20.ipynb` generates a puppy prompt in Madhubani, line-art, Pokémon, and concept-art styles, then compares ordinary generation with saturation-guided generation. The custom objective demonstrates how gradients can steer image properties while retaining the pretrained model.

An accompanying interface is available on [Hugging Face Spaces](https://huggingface.co/spaces/ToletiSri/TSAI_S20).
