# Traffic RL Simulator Frontend

This app is a React + Vite traffic simulator with integrated RL model training, evaluation, and runtime control.

## Run locally

1. Start backend (from `../backend`):
   - `uvicorn main:app --reload --port 8000`
2. Start frontend (from this folder):
   - `npm install`
   - `npm run dev`

## Training + Evaluation workflow

1. Open the app and switch **Mode** to **RL**.
2. Pick a **Training Preset**:
   - `Quick Demo`: fast iterations for live demo.
   - `Balanced`: default hackathon setting.
   - `Thorough`: slower but stronger training run (always runs full 350 episodes, no early stop).
3. Set **Training Seed** for reproducibility.
4. Click **Start Training**.
5. Monitor live progress:
   - episode / epsilon
   - reward
   - queue
6. When training completes:
   - model is saved to local storage
   - model metadata is shown in panel
   - benchmark runs automatically (Fixed vs Adaptive vs RL)
7. Use lifecycle controls:
   - **Download Model**: export `.json` weights
   - **Load model**: import previously exported weights
   - **Run Benchmark**: re-run validation with current model
   - **Clear Saved Model**: remove model from local storage

## RL runtime control

You can choose where RL decisions are computed:

- **Frontend**: browser-based inference.
- **Backend**: server-side inference through `/signal/tick` with `strategy=rl`.

To use backend inference:

1. Train or load a model in frontend.
2. Click **Upload Model To Backend**.
3. Set **RL Inference Source** to **Backend**.

## Benchmark metrics tracked

Validation reports these metrics across seeded episodes:

- average queue length
- total switches
- average reward per episode
- vehicles cleared (throughput)

## Engineering quality checks

Run these before demo:

- Frontend build: `npm run build`
- Frontend RL tests: `npm run test:rl`
- Backend signal logic tests (from `../backend`):
  - `python -m unittest test_signal_logic.py`
