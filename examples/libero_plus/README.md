# LIBERO Benchmark

This example runs the LIBERO benchmark: https://github.com/Lifelong-Robot-Learning/LIBERO

Note: When updating requirements.txt in this directory, there is an additional flag `--extra-index-url https://download.pytorch.org/whl/cu113` that must be added to the `uv pip compile` command.

This example requires git submodules to be initialized. Don't forget to run:

```bash
git submodule update --init --recursive
```

## With Docker (recommended)

```bash
# Grant access to the X11 server:
sudo xhost +local:docker

# To run with the default checkpoint and task suite:
SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build

# To run with glx for Mujoco instead (use this if you have egl errors):
MUJOCO_GL=glx SERVER_ARGS="--env LIBERO" docker compose -f examples/libero/compose.yml up --build
```

You can customize the loaded checkpoint by providing additional `SERVER_ARGS` (see `scripts/serve_policy.py`), and the LIBERO task suite by providing additional `CLIENT_ARGS` (see `examples/libero/main.py`).
For example:

```bash
# To load a custom checkpoint (located in the top-level openpi/ directory):
export SERVER_ARGS="--env LIBERO policy:checkpoint --policy.config pi05_libero --policy.dir ./my_custom_checkpoint"

# To run the libero_10 task suite:
export CLIENT_ARGS="--args.task-suite-name libero_10"
```

## Without Docker (not recommended)

7 Perturbation Dimensions

- Objects Layout - Confounding objects and target object displacement
- Camera Viewpoints - Position, orientation, and field-of-view changes
- Robot Initial States - Manipulator initial pose variations
- Language Instructions - LLM-based instruction rewriting
- Light Conditions - Intensity, direction, color, and shadow variations
- Background Textures - Scene and surface appearance changes
- Sensor Noise - Photometric distortions and image degradation

Terminal window 1:

```bash
# New dependencies installed on top of LIBERO
apt install libexpat1
apt install libfontconfig1-dev
apt install libpython3-stdlib
apt-get install libmagickwand-dev

# Create virtual environment
uv venv --python 3.8 examples/libero_plus/.venv
source examples/libero_plus/.venv/bin/activate
uv pip sync examples/libero_plus/requirements.txt third_party/libero/requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e /inspire/hdd/project/wuliqifa/chenxinyan-240108120066/chaiyilin/RoboTwin/policy/Representation-Matters/LIBERO-plus/
uv pip install -r /inspire/hdd/project/wuliqifa/chenxinyan-240108120066/chaiyilin/RoboTwin/policy/Representation-Matters/LIBERO-plus/extra_requirements.txt
export PYTHONPATH=$PYTHONPATH:/inspire/ssd/project/wuliqifa/zhousizhuo-240107010006/moe-vla/LIBERO-plus

# Run the simulation
xvfb-run -a env MUJOCO_GL=glx python examples/libero_plus/main.py \
  --args.task-suite-name libero_object \
  --args.category "Objects Layout" \
  --args.video_out_path data/libero_objects_layout/ \
  --args.num_trials_per_task 1 \
  --args.results-json-path out/libero_objects_layout.json \
  --args.port 8001

xvfb-run -a env MUJOCO_GL=glx python examples/libero_plus/main.py \
  --args.task-suite-name libero_object \
  --args.category "Light Conditions" \
  --args.video_out_path data/libero_light_conditions/ \
  --args.num_trials_per_task 1 \
  --args.results-json-path out/libero_light_conditions.json \
  --args.port 8002

python examples/libero_plus/main.py \
  --args.task-suite-name libero_object \
  --args.category "Sensor Noise" \
  --args.video_out_path data/libero_sensor_noise_pi0_libero_object_no_control_60k \
  --args.num_trials_per_task 1 \
  --args.results-json-path out/libero_sensor_noise_pi0_libero_object_no_control_60k.json \
  --args.port 8081

# To run with glx for Mujoco instead (use this if you have egl errors):
xvfb-run -a env MUJOCO_GL=glx python examples/libero_plus/main.py
```

Terminal window 2:

```bash
# Run the server
# for sii machine
unset LD_LIBRARY_PATH 
CUDA_VISIBLE_DEVICES=1 uv run --no-sync scripts/serve_policy.py \
  --env LIBERO \
  --port 8041 \
  policy:checkpoint \
  --policy.config pi0_libero_object_no_control \
  --policy.dir checkpoints/pi0_libero_object_no_control/pi0_libero_object_no_control_cst_lr/60000
```

## Multiple Task Evalutation

Run:
```bash
unset LD_LIBRARY_PATH 
uv run --no-sync scripts/eval_libero_plus_all.py \
  --args.policy.python .venv/bin/python \
  --args.policy.config pi0_libero_object_gripper_depth_skill_no_control \
  --args.policy.checkpoint_dir checkpoints/pi0_libero_object_gripper_depth_skill_no_control/pi0_libero_object_gripper_depth_skill_no_control/60000 \
  --args.policy.port 8000 \
  --args.policy.cuda-visible-devices 0,1,2,3 \
  --args.eval.python examples/libero_plus/.venv/bin/python \
  --args.eval.pythonpath /inspire/ssd/project/wuliqifa/zhousizhuo-240107010006/moe-vla/LIBERO-plus \
  --args.eval.max-workers 7
```

## Results

If you want to reproduce the following numbers, you can evaluate the checkpoint at `gs://openpi-assets/checkpoints/pi05_libero/`. This
checkpoint was trained in openpi with the `pi05_libero` config.

| Model | Libero Spatial | Libero Object | Libero Goal | Libero 10 | Average |
|-------|---------------|---------------|-------------|-----------|---------|
| π0.5 @ 30k (finetuned) | 98.8 | 98.2 | 98.0 | 92.4 | 96.85
