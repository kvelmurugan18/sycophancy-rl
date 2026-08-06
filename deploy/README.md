# Deployment layout

- `trainer/` pins dependencies for the isolated GPU trainer image.
- `kaggle/` contains the staged script runner and Kaggle dependency lock.
- `env-server/` packages the local diagnostic FastAPI service.
- `demo-space/` packages the optional Gradio diagnostic client.

The GPU trainer and diagnostic API deliberately use separate images and Compose
files. Deployment directories own runtime-specific configuration only;
scientific behavior stays in `src/sycophancy_rl`.
