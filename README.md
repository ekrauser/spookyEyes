# spookyeyes

Two round 240x240 GC9A01A TFTs on a Raspberry Pi 3B+, animated as a pair of
eyes that wander, blink, dilate — and snap wide open when a PIR sensor sees a
trick-or-treater. Controlled over MQTT with Home Assistant auto-discovery.
Three themes ship in `themes/`: `human`, `demon`, `ghost`.

The displays are driven by the mainline `panel-mipi-dbi` kernel driver (a
custom firmware blob carries the GC9A01 init sequence), so the app just writes
RGB565 frames to `/dev/fb1` and `/dev/fb2` — no userspace SPI bit-banging.

## Hardware

- Raspberry Pi 3B+ (any 40-pin Pi works; pin numbers below are BCM + physical)
- 2x 1.28" round GC9A01A 240x240 TFT (Adafruit #6178 "with EYESPI Connector",
  each on an Adafruit EYESPI Breakout Board #5613 + EYESPI FPC cable — wire to
  the breakout's silkscreen labels; the display CS is labelled **TCS**. Plain
  pin-header GC9A01A modules work identically)
- PIR motion sensor, 3.3 V-logic output (HC-SR501 or Adafruit PIR)
- Micro SD card (8 GB+), 5 V / 2.5 A supply, jumper wires

## Wiring

Both displays share the SPI0 bus (SCK + MOSI); each has its own chip select,
D/C and reset. **All grounds must be common** — Pi and both panels (and the
PIR), even if anything is powered separately. Missing common ground is the
number-one cause of "nothing works / garbage pixels".

### Displays (SPI0 harness)

| Display pin | Left eye → Pi          | Right eye → Pi         |
|-------------|------------------------|------------------------|
| Vin         | 3V3 (phys 1)           | 3V3 (phys 17)          |
| GND         | GND (phys 6)           | GND (phys 9)           |
| SCK         | GPIO11 / SCLK (phys 23) — shared | same (shared)          |
| MOSI        | GPIO10 / MOSI (phys 19) — shared | same (shared)          |
| TCS (CS)    | GPIO8 / CE0 (phys 24)  | GPIO7 / CE1 (phys 26)  |
| DC          | GPIO25 (phys 22)       | GPIO24 (phys 18)       |
| RST         | GPIO27 (phys 13)       | GPIO23 (phys 16)       |

Leave **Lite** (backlight is on by default), **MISO**, **SDCS**, and **EYESPI
connector pins 11–18** unconnected. The panels are written to only
(`write-only` in the overlay), so MISO is never used.

### PIR sensor

| PIR pin | Pi                              |
|---------|---------------------------------|
| VCC     | 5V (phys 2)                     |
| OUT     | GPIO17 (phys 11)                |
| GND     | GND (phys 14)                   |

HC-SR501 outputs 3.3 V despite 5 V supply — safe to connect directly. Mount
the PIR at least ~30 cm from the Pi; the SoC's heat plume and WiFi bursts
cause false triggers at point-blank range.

## Flash the SD card

1. Raspberry Pi Imager → Device: Raspberry Pi 3 → OS: **Raspberry Pi OS Lite
   (64-bit)** (Trixie) → your SD card.
2. In the customisation dialog (Edit Settings) set:
   - hostname: `spookyeyes`
   - enable SSH (password or key)
   - your username + password, WiFi SSID/password, locale
3. Write, boot the Pi, then:

```sh
ssh <you>@spookyeyes.local
sudo apt update && sudo apt full-upgrade -y
```

## Bring-up, staged

Do these in order — each stage isolates a different class of problem
(wiring → kernel driver → app).

### Stage A — wiring smoke test (userspace SPI, Adafruit library)

Uses `spidev` + Blinka, no custom firmware involved, so any failure here is
wiring, not software.

```sh
# Enable SPI0 with NO hardware chip-select lines. The smoke test toggles CS
# itself, and the modern lgpio GPIO backend cannot claim CE0/CE1 while the
# kernel owns them (you'd get "lgpio.error: 'GPIO busy'"). Don't use
# raspi-config / dtparam=spi=on here — remove that line if it's present.
echo 'dtoverlay=spi0-0cs' | sudo tee -a /boot/firmware/config.txt
sudo reboot
```

```sh
sudo apt install -y python3-venv python3-dev swig liblgpio-dev
python3 -m venv ~/smoke
~/smoke/bin/pip install adafruit-circuitpython-rgb-display
~/smoke/bin/pip uninstall -y RPi.GPIO
~/smoke/bin/pip install rpi-lgpio
```

(`python3-dev`, `swig`, and `liblgpio-dev` are needed because piwheels often
lags a fresh OS's Python version, so pip builds `lgpio`/`RPi.GPIO` from
source — the lgpio wrapper also links against the system C library. `RPi.GPIO` is
pulled in by Blinka but must be replaced by the `rpi-lgpio` shim — install the
shim only *after* uninstalling the original; both ship the same module path.)

Left eye — these are exactly the pins Adafruit's published Python example for
this display uses, so it doubles as a sanity check against their guide:

```sh
~/smoke/bin/python - <<'EOF'
import board, digitalio
from adafruit_rgb_display import gc9a01a
disp = gc9a01a.GC9A01A(
    board.SPI(),
    cs=digitalio.DigitalInOut(board.CE0),
    dc=digitalio.DigitalInOut(board.D25),
    rst=digitalio.DigitalInOut(board.D27),
    baudrate=12_000_000,  # deliberately slow: this stage tests wiring, not speed
)
disp.fill(0xF800)  # left eye: solid red
EOF
```

Right eye — same script with the three pin edits (`CE1`, `D24`, `D23`):

```sh
~/smoke/bin/python - <<'EOF'
import board, digitalio
from adafruit_rgb_display import gc9a01a
disp = gc9a01a.GC9A01A(
    board.SPI(),
    cs=digitalio.DigitalInOut(board.CE1),
    dc=digitalio.DigitalInOut(board.D24),
    rst=digitalio.DigitalInOut(board.D23),
    baudrate=12_000_000,  # deliberately slow: this stage tests wiring, not speed
)
disp.fill(0x001F)  # right eye: solid blue
EOF
```

Both panels light up solid? Wiring is good — continue. (If Blinka complains
about GPIO access on Trixie, the `rpi-lgpio` package installed above is the
fix; make sure plain `RPi.GPIO` is *not* also installed in the venv.)

### Stage B — kernel driver + framebuffers

Get the repo onto the Pi (see Stage C for rsync) or just fetch the three
files `pi/gc9a01.txt`, `pi/build_firmware.sh`, `pi/config.txt.snippet`. Then:

```sh
cd ~/spookyEyes
bash pi/build_firmware.sh          # compiles gc9a01.txt -> /lib/firmware/gc9a01.bin
```

Edit `/boot/firmware/config.txt`: **remove the `dtoverlay=spi0-0cs` line**
added in Stage A (and any `dtparam=spi=on`), and append the contents of
`pi/config.txt.snippet` (two
`mipi-dbi-spi` stanzas: left eye on CE0 with DC=25/RST=27, right on CE1 with
DC=24/RST=23, 40 MHz, write-only). Then:

```sh
sudo reboot
```

After reboot:

```sh
dmesg | grep -i mipi               # expect two panel-mipi-dbi probes, no errors
ls /dev/fb*                        # expect two fb nodes (+ fb0 only if HDMI is attached)
sudo apt install -y python3-numpy
sudo python3 pi/test_pattern.py    # red/white "L" bullseye left, blue/white "R" right
                                   # (sudo only needed until install.sh adds you to the video group)
```

Which panel gets `/dev/fb1` vs `/dev/fb2` follows **async probe order and can
swap between boots** — that's why the app and `test_pattern.py` address panels
by stable SPI address instead (`spi0.0` = CE0 = left, `spi0.1` = CE1 = right)
and resolve the fb number through sysfs at startup.

If "L" and "R" come up on the wrong panels, swap the `fb_left`/`fb_right`
values in `config.toml` later (or swap the CE0/CE1 wires).

### Stage C — deploy the app

From your dev machine:

```sh
rsync -av --exclude .git --exclude .venv --exclude __pycache__ \
    ./ <you>@spookyeyes.local:~/spookyEyes/
```

On the Pi:

```sh
cd ~/spookyEyes
bash pi/install.sh
```

`install.sh` is idempotent (re-run it after every rsync): installs
`python3-venv`/`git`, creates `~/spookyeyes-venv`, `pip install -e
".[mqtt,pir]"`, creates `config.toml` from the example (with `output = "fb"`),
and installs + enables `spookyeyes.service` (systemd, `Restart=always`,
starts after network-online).

Edit `~/spookyEyes/config.toml`: set `[mqtt] enabled = true` with your broker
host/credentials, `[pir] enabled = true` if the sensor is wired, pick the
`[theme]`. Then:

```sh
sudo systemctl start spookyeyes
journalctl -u spookyeyes -f        # logs measured FPS every 5 s
```

## Development on a PC

No hardware needed — there is a pygame preview:

```sh
git clone <this repo> && cd spookyEyes
python3 -m venv .venv
.venv/bin/pip install -e ".[preview,dev]"
.venv/bin/spookyeyes --output preview                 # live window, both eyes
.venv/bin/spookyeyes --output preview --theme demon --seed 42
.venv/bin/spookyeyes --output record --frames 150 --gif out/demo.gif
.venv/bin/python tools/gen_art.py                     # regenerate theme art
.venv/bin/pytest -q
```

`--frames N` exits after N frames (benchmarks/tests), `--seed` makes the
behavior deterministic.

## MQTT + Home Assistant

Topics (base topic configurable, default `spookyeyes`):

| Topic                        | Dir       | Payload                                |
|------------------------------|-----------|----------------------------------------|
| `spookyeyes/cmd/theme`       | → device  | `human` \| `demon` \| `ghost`          |
| `spookyeyes/cmd/mode`        | → device  | `idle` \| `scare` \| `stare` \| `sleep`|
| `spookyeyes/cmd/brightness`  | → device  | float `0.0`–`1.0`                      |
| `spookyeyes/cmd/blink`       | → device  | any payload → one blink                |
| `spookyeyes/state/theme`     | ← device  | retained, current theme                |
| `spookyeyes/state/mode`      | ← device  | retained, current mode                 |
| `spookyeyes/state/brightness`| ← device  | retained, current brightness           |
| `spookyeyes/availability`    | ← device  | retained `online` / `offline` (LWT)    |

`scare` runs the startle animation (~6 s) and returns to `idle` by itself.
The PIR triggers the same thing locally, rate-limited by `[pir] cooldown`.

**Home Assistant:** with `[mqtt] discovery = true` (the default) and the MQTT
integration set up in HA, a "Spooky Eyes" device appears automatically with
`select` entities for theme and mode, a `number` for brightness, and a
`button` for blink — nothing to configure. Availability tracks the service.

Example automation — front-door motion triggers a scare during the evening:

```yaml
alias: Spooky eyes scare at the front door
triggers:
  - trigger: state
    entity_id: binary_sensor.front_door_motion
    to: "on"
conditions:
  - condition: time
    after: "17:00:00"
    before: "23:00:00"
actions:
  - action: mqtt.publish
    data:
      topic: spookyeyes/cmd/mode
      payload: scare
mode: single
```

## Troubleshooting

- **No `/dev/fb1`/`/dev/fb2`** — `dmesg | grep -iE 'mipi|gc9a01'`. Driver not
  probing: config.txt stanzas missing/typo'd, or a stray Stage-A SPI line
  (`dtparam=spi=on` or `dtoverlay=spi0-0cs`) still claiming SPI0. Probing but failing: `ls -la /lib/firmware/gc9a01.bin`
  — if missing, run `pi/build_firmware.sh` and reboot.
- **Colors wrong** — Negative image: the panel needs inversion; make sure
  `command 0x21` (INVON) is present in `pi/gc9a01.txt` (some clones want it
  removed instead). Red/blue swapped or image mirrored/rotated: change the
  MADCTL value (`command 0x36 0x48`) — try `0x18`, `0x28`, `0x48`, `0x88`.
  After any edit: `bash pi/build_firmware.sh && sudo reboot`.
- **Panels probe fine but stay frozen on old content** — the DRM pipeline
  behind the fbdev is disabled (`enable=0` in
  `/sys/kernel/debug/dri/*/state`): on headless boots fbcon never takes over
  these framebuffers, so nothing performs the initial mode-set. The app and
  `pi/test_pattern.py` force one automatically on open
  (FBIOPUT_VSCREENINFO + FB_ACTIVATE_FORCE); if you write to `/dev/fbN` with
  other tools, they must do the same.
- **Glitches / tearing / random pixels** — Drop SPI to 32 MHz (commented
  fallback in `pi/config.txt.snippet`, change **both** stanzas), pin the core
  clock (`core_freq=400` + `core_freq_min=400`, also in the snippet), shorten
  the SPI leads (< 15 cm ideally), and re-check the **common ground**.
- **PIR fires constantly** — It is too close to the Pi (heat + WiFi):
  relocate it, shield its underside, turn its sensitivity pot down, and raise
  `[pir] cooldown` in config.toml. HC-SR501s also self-trigger for the first
  ~60 s after power-on; that is normal.
- **Low FPS** — Confirm the *kernel* driver is in use, not a userspace/spidev
  path: `dmesg | grep panel-mipi-dbi` must show both panels, and `/dev/spidev0.*`
  should **not** exist (if it does, a Stage-A SPI line — `dtparam=spi=on` or
  `dtoverlay=spi0-0cs` — is still in config.txt).
  Check `journalctl -u spookyeyes` for the measured FPS line. Expectations:
  the app renders at `fps = 30`, but two full-frame 240x240 RGB565 panels on a
  shared 40 MHz bus top out around ~21 fps each of *panel* refresh — the
  kernel's deferred flushing simply drops intermediate frames, which the
  animations are tuned to tolerate. Set `[display] fps = 20` if you'd rather
  save the CPU.
- **Service starts before the network / MQTT** — it retries: the unit has
  `Restart=always`/`RestartSec=3` and the MQTT client auto-reconnects, so a
  briefly-unreachable broker only delays the entities going `online`.
