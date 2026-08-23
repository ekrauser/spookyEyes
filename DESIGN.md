# spookyeyes — design contract

Dual 240x240 round GC9A01A displays driven from a Raspberry Pi 3B+ as animated
Halloween eyes. This document is the **contract** all modules are written against.
`src/spookyeyes/model.py` and `src/spookyeyes/config.py` are fully implemented and
authoritative for shared types; do not change their field names.

## Data flow

```
inputs/mqtt.py ──┐                                        ┌─> outputs/fb.py      (/dev/fb1 + /dev/fb2, RGB565)
inputs/pir.py ───┤→ queue.Queue[Event] → app.py main loop ├─> outputs/preview.py (pygame window, dev machine)
                 │        │                    │          ├─> outputs/record.py  (PNG frames / GIF)
                 │        v                    v          └─> outputs/null.py    (benchmarks)
                 │  behavior.BehaviorEngine  eye.EyeRenderer (x2, left+right)
                 │  .step(dt) -> (EyeState, EyeState)
                 └─ app handles "theme" events itself: reload Theme, rebuild renderers,
                    behavior.set_motion(theme.motion)
```

- The app runs a fixed-timestep loop at `display.fps` (default 30) using a
  monotonic clock; it drains the event queue each tick, calls
  `behavior.handle(event)` for everything except `"theme"`/`"quit"`, steps the
  behavior engine, renders both eyes, applies optional horizontal mirror
  (`np.fliplr`) per config, and calls `output.show(left, right)`.
- Frames are numpy `uint8` arrays of shape `(240, 240, 3)`, RGB order.
- All threads communicate with the loop **only** via the `Event` queue.
  `MqttInput.publish_state(...)` is called *from* the app thread after changes.

## Module contracts (file ownership per implementation agent)

### themes.py + eye.py (+ tests/test_render.py, tests/conftest.py) — agent RENDER

`themes.load_theme(themes_dir: Path, name: str) -> Theme`

`Theme` is a dataclass holding parsed `theme.json` plus loaded PIL images:
`name, path, background (RGB tuple), sclera (PIL RGB, square >= 240, typically 320),
iris (PIL RGBA), highlight (PIL RGBA or None), pupil_shape ("round"|"slit"|"none"),
pupil_color (RGB), pupil_min_frac, pupil_max_frac, slit_width_frac, iris_frac,
gaze_range_px, sclera_parallax, lid_color (RGB), lid_style ("curved"|"straight"),
lid_upper_bias, motion (model.MotionParams)`.
Missing optional art (highlight) → None. Missing/invalid required art or JSON → raise
`ThemeError` (define in themes.py) with a helpful message. Unknown JSON keys ignored.

`eye.EyeRenderer(theme: Theme)` — pre-bakes at construction:
- `PUPIL_STEPS = 12` iris sprites: iris RGBA composited with the pupil drawn at
  dilation fraction lerp(min_frac, max_frac, k/(STEPS-1)). Slit pupils are vertical
  ellipses (width = slit_width_frac * iris * dilation-lerped scale, height = iris);
  round pupils are centered circles; "none" bakes no pupil.
- `LID_STEPS = 24` eyelid alpha masks (uint8 L-mode 240x240) quantizing
  openness 0..1.25. Style "curved": lids are circle-arc segments (upper lid comes
  down `upper_bias` of the closure, lower lid up the rest); "straight": chords.
- a round vignette alpha mask (opaque circle d=240) — outside the circle is black.

`EyeRenderer.render(state: EyeState) -> np.ndarray (240,240,3) uint8`:
1. crop 240x240 window from sclera, offset by `-gaze_px * sclera_parallax`
2. alpha-composite pre-baked iris sprite `k = round(state.pupil * (PUPIL_STEPS-1))`
   centered at `center + gaze_px` where
   `gaze_px = (gaze_x * gaze_range_px, -gaze_y * gaze_range_px)` (screen y down)
3. alpha-composite highlight at a fixed offset from iris center, counter-moving
   `-0.15 * gaze_px` (specular stays roughly light-locked)
4. composite lid color over the frame using the lid mask for `state.openness`
5. apply vignette; scale by `state.brightness` if < 1.0 (numpy, not PIL)
No per-frame `resize`/`rotate` calls — perf budget is <= 8 ms per eye on a Pi 3
(use PIL `Image.paste`/`alpha_composite` on prebaked sprites + one numpy pass).

`tests/conftest.py` must provide a `fixture_theme(tmp_path)`-style pytest fixture
that programmatically writes a minimal valid theme dir (solid-color PNGs via PIL +
theme.json) so all tests run without the real `themes/` assets.

### behavior.py (+ tests/test_behavior.py) — agent BEHAVIOR

`BehaviorEngine(motion: MotionParams, rng: random.Random | None = None)`
- `step(dt: float) -> tuple[EyeState, EyeState]` — advance and return (left, right)
- `handle(event: Event) -> None` — kinds: mode, brightness, blink, motion
- `set_motion(motion: MotionParams) -> None` — on theme switch, keep pose
- `mode` property -> Mode

State machine (all randomness through `self.rng` for testability):
- **IDLE**: pick gaze targets within `wander` radius at uniform
  `saccade_interval`; move with ease-out cubic over a `saccade_duration` sample;
  blinks at uniform `blink_interval` (10% double-blink); pupil drifts smoothly
  (bounded random walk, rate `pupil_speed`); `drift` adds slow sinusoidal motion;
  `flicker` multiplies brightness by `1 - flicker * |noise|` per tick.
- Eye independence: right eye target = left target + `crazy`-scaled independent
  offset (clamped to valid range). `crazy=0` → identical.
- **SCARE** (entered on `motion` when IDLE, or mode event): openness → 1.25 fast,
  pupil → 0.1 fast, gaze snaps to center-front then 2–3 rapid saccades; after
  ~6 s auto-return to IDLE. Re-triggering while active restarts the timer.
  `motion` events are ignored in STARE/SLEEP and rate-limited by inputs anyway.
- **STARE**: gaze eases to (0,0) and holds; blink interval x4.
- **SLEEP**: openness eases to 0 and holds; every ~10 s a small lid twitch;
  on leaving, eyes reopen smoothly.
- Blink envelope: close in ~0.08 s / `blink_speed`, open in ~0.15 s / `blink_speed`;
  blink multiplies the mode's base openness (so a blink in SCARE closes from 1.25).
- All outputs clamped to documented EyeState ranges; `step` must be deterministic
  given a seeded rng and fixed dt sequence (tests rely on this).

### app.py + outputs/ (+ tests/test_outputs.py, tests/test_app.py) — agent APP

`outputs/__init__.py`: `Output` protocol (`show(left, right)`, `close()`) and
`make_output(cfg: DisplayConfig) -> Output` factory (imports preview/pygame lazily).
`outputs/fb.py`:
- `to_rgb565(frame) -> np.ndarray dtype '<u2' shape (240,240)` (vectorized)
- `FramebufferOutput(paths: tuple[str, str])` — open + mmap both fbdev devices,
  write RGB565 little-endian; size mismatches raise with a clear message; if the
  device reports a larger virtual size, write into the top-left 240x240 stride-aware.
`outputs/record.py`: `RecordOutput(dir, gif_path=None, max_frames=None, every=1)` —
saves side-by-side composite PNGs `frame_00000.png`... and optionally an animated
GIF on close (PIL, ~15 fps playback).
`outputs/preview.py`: `PreviewOutput(scale=2)` — pygame window showing both eyes
side by side with a small gap; window close/ESC pushes `Event("quit")` if given a
queue (constructor arg `events: queue.Queue | None = None`); pump events each show.
`outputs/null.py`: already implemented — leave as is.

`app.py`: `main(argv=None)` CLI: `--config PATH`, `--theme NAME`,
`--output {preview,fb,record,null}`, `--frames N` (exit after N frames, for tests
and benchmarks), `--record-dir`, `--gif PATH`, `--fps N`, `--seed N` (deterministic
behavior for tests/demos), `--verbose`. CLI overrides config. Wires config →
theme load → renderers → behavior → inputs (mqtt/pir only if enabled; import
errors for optional deps produce a warning, not a crash) → output; runs the loop;
logs measured FPS every 5 s (`logging`, INFO); SIGINT/SIGTERM → clean close.
On `"theme"` event: `themes.load_theme`, rebuild both renderers,
`behavior.set_motion`, then `mqtt.publish_state` if mqtt is up.
`tests/test_app.py` may exercise the full loop headlessly
(`--output null --frames 30`) via `main([...])` — it is expected to pass only
once RENDER and BEHAVIOR land; keep it import-light so pytest collects it anyway.

### inputs/ (+ tests/test_inputs.py) — agent INPUTS

`inputs/mqtt.py`: `MqttInput(cfg: MqttConfig, events: queue.Queue)` using paho-mqtt
v2 API (CallbackAPIVersion.VERSION2), lazy import so the package works without it.
- `start()` — background network thread (`loop_start`), auto-reconnect
- topics (base = cfg.base_topic):
  - subscribe `base/cmd/theme` (payload: theme name) → Event("theme", name)
  - subscribe `base/cmd/mode` (idle|scare|stare|sleep) → Event("mode", value)
  - subscribe `base/cmd/brightness` (float 0..1) → Event("brightness", value)
  - subscribe `base/cmd/blink` (any payload) → Event("blink")
  - publish retained availability `base/availability` = "online", LWT "offline"
  - `publish_state(theme: str, mode: str, brightness: float)` publishes retained
    `base/state/theme`, `base/state/mode`, `base/state/brightness`
- if `cfg.discovery`: on connect publish retained Home Assistant MQTT discovery
  configs under `homeassistant/select/spookyeyes_theme/config` (options
  human/demon/ghost), `homeassistant/select/spookyeyes_mode/config`,
  `homeassistant/number/spookyeyes_brightness/config` (0..1 step 0.05),
  `homeassistant/button/spookyeyes_blink/config`, all sharing one device block
  (identifiers ["spookyeyes"], name "Spooky Eyes") with availability_topic set.
- malformed payloads: log warning, drop. Never raise into paho callbacks.
`inputs/pir.py`: `PirInput(cfg: PirConfig, events)` via gpiozero MotionSensor
(lazy import; on non-Pi hosts construction raises `PirUnavailable` (define it) —
the app treats that as a warning). `when_motion` → Event("motion") but only if
`cooldown` has elapsed since the last forwarded event (time.monotonic).
Tests: fake the paho client via injection (`MqttInput(..., client_factory=...)`)
and test topic → Event mapping, discovery payload shape (json), cooldown logic of
PirInput with a stubbed sensor class (`sensor_factory` injection).

### tools/gen_art.py + themes/{human,demon,ghost}/ — agent ART

Procedurally generates all theme art with PIL/numpy (no external assets, no
network): for each theme write `themes/<name>/theme.json` + `sclera.png` (RGB 320,
must look good when cropped ±gaze_range_px*sclera_parallax) + `iris.png`
(RGBA, diameter = iris_frac*240 px canvas) + `highlight.png` (RGBA).
- **human**: off-white sclera w/ subtle red veins + shading toward edges; amber-to
  -brown ringed iris with radial fiber streaks; round pupil; soft white highlight;
  natural motion (crazy≈0.05, wander 0.35).
- **demon**: blackened-red sclera w/ dark veins; fiery ember iris (yellow core →
  deep red rim, glow halo baked into the RGBA edge); "slit" pupil; motion:
  faster saccades, crazy≈0.15, high flicker 0.
- **ghost**: near-black smoky sclera; pale cyan-white glowing orb iris with wide
  soft alpha falloff (glow baked in); pupil "none" or tiny dark core; motion:
  drift high, slow saccades, flicker≈0.25, blink rare.
`python tools/gen_art.py [--theme NAME] [--out themes/]` regenerates; commit the
PNGs. Verify visually: render each layer + save a composite mock (sclera + iris
pasted center) per theme to `out/art_check_<name>.png`, then LOOK at those PNGs
(Read tool) and iterate until they genuinely read as creepy/theme-appropriate.
theme.json must match the schema below exactly.

### pi/ + README.md — agent PI (needs WebFetch)

See plan (approved): `pi/gc9a01.txt` (mipi-dbi-cmd init source — transcribe the
GC9A01 init sequence by fetching carlfriess/GC9A01_demo GC9A01.c and the syntax
from notro/panel-mipi-dbi; cross-check RPi forum t=365153), `pi/build_firmware.sh`,
`pi/config.txt.snippet` (two mipi-dbi-spi stanzas, speed=40000000, left:
dc-gpio=25/reset-gpio=27, right: dc-gpio=24/reset-gpio=23), `pi/test_pattern.py`
(standalone numpy → both /dev/fb*, distinct patterns), `pi/install.sh` (idempotent:
apt python3-venv/git, venv at ~/spookyeyes-venv, pip install -e
.[mqtt,pir], install+enable `pi/spookyeyes.service` running
`spookyeyes --config /home/<user>/spookyEyes/config.toml --output fb`),
`pi/spookyeyes.service`, and a complete README.md (wiring tables incl. PIR, flash
walkthrough, staged bring-up with the Blinka smoke test first, firmware build,
config.txt, deploy via rsync, MQTT/HA usage incl. example automation YAML,
troubleshooting from the research caveats: common ground, 40→32 MHz fallback,
core_freq pinning, INVON/MADCTL tweaks, PIR-near-Pi false triggers).

## theme.json schema (authoritative example)

```json
{
  "name": "human",
  "background": [10, 5, 5],
  "layers": {"sclera": "sclera.png", "iris": "iris.png", "highlight": "highlight.png"},
  "pupil": {"shape": "round", "color": [5, 0, 0], "min_frac": 0.22, "max_frac": 0.55,
            "slit_width_frac": 0.14},
  "iris_frac": 0.62,
  "gaze_range_px": 52,
  "sclera_parallax": 0.35,
  "eyelids": {"color": [8, 4, 4], "style": "curved", "upper_bias": 0.6},
  "motion": {"saccade_interval": [0.6, 3.0], "saccade_duration": [0.09, 0.22],
             "wander": 0.35, "crazy": 0.1, "blink_interval": [2.0, 8.0],
             "blink_speed": 1.0, "pupil_speed": 0.5, "drift": 0.0, "flicker": 0.0}
}
```
`layers.highlight` optional. Fractions are relative to iris diameter
(`iris_frac` relative to 240). Motion keys map 1:1 to `model.MotionParams`.

## Conventions

- Python 3.11+, type hints everywhere, `from __future__ import annotations`.
- stdlib `logging` (`logging.getLogger("spookyeyes.<module>")`), no prints outside
  `tools/` and `pi/`.
- Run tests with `.venv/bin/pytest tests/test_<yours>.py -q` before finishing.
- Do not modify files outside your ownership list; model.py/config.py are frozen.
- Optional deps (pygame, paho-mqtt, gpiozero) must be imported lazily so
  `import spookyeyes` and the null/record paths work with numpy+pillow only.
