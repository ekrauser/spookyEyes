"""Tests for the output backends (fb / record / preview / factory)."""

from __future__ import annotations

import queue

import numpy as np
import pytest
from PIL import Image

from spookyeyes.config import DisplayConfig
from spookyeyes.model import SIZE
from spookyeyes.outputs import Output, make_output
from spookyeyes.outputs.fb import FramebufferOutput, to_rgb565
from spookyeyes.outputs.null import NullOutput
from spookyeyes.outputs.record import RecordOutput


def _frame(rng_seed: int) -> np.ndarray:
    rng = np.random.default_rng(rng_seed)
    return rng.integers(0, 256, (SIZE, SIZE, 3), dtype=np.uint8)


def _px(rgb: tuple[int, int, int]) -> int:
    frame = np.full((1, 1, 3), rgb, dtype=np.uint8)
    return int(to_rgb565(frame)[0, 0])


# --------------------------------------------------------------------------- #
# to_rgb565
# --------------------------------------------------------------------------- #


def test_to_rgb565_known_values() -> None:
    assert _px((255, 0, 0)) == 0xF800  # pure red -> top 5 bits
    assert _px((0, 255, 0)) == 0x07E0  # pure green -> middle 6 bits
    assert _px((0, 0, 255)) == 0x001F  # pure blue -> bottom 5 bits
    assert _px((255, 255, 255)) == 0xFFFF
    assert _px((0, 0, 0)) == 0x0000


def test_to_rgb565_dtype_shape_and_little_endian_bytes() -> None:
    frame = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    frame[..., 0] = 255  # all red
    out = to_rgb565(frame)
    assert out.dtype == np.dtype("<u2")
    assert out.shape == (SIZE, SIZE)
    raw = out.tobytes()
    # 0xF800 little-endian: low byte 0x00 first, high byte 0xF8 second.
    assert raw[:2] == b"\x00\xf8"

    green = np.zeros((1, 1, 3), dtype=np.uint8)
    green[..., 1] = 255
    assert to_rgb565(green).tobytes() == b"\xe0\x07"  # 0x07E0 LE
    white = np.full((1, 1, 3), 255, dtype=np.uint8)
    assert to_rgb565(white).tobytes() == b"\xff\xff"


def test_to_rgb565_mid_gray_roundtrips_within_tolerance() -> None:
    val = _px((128, 128, 128))
    r5 = (val >> 11) & 0x1F
    g6 = (val >> 5) & 0x3F
    b5 = val & 0x1F
    # Standard bit-replication expansion back to 8 bits.
    r8 = (r5 << 3) | (r5 >> 2)
    g8 = (g6 << 2) | (g6 >> 4)
    b8 = (b5 << 3) | (b5 >> 2)
    assert abs(r8 - 128) <= 8
    assert abs(g8 - 128) <= 4
    assert abs(b8 - 128) <= 8


def test_to_rgb565_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        to_rgb565(np.zeros((SIZE, SIZE, 3), dtype=np.float32))
    with pytest.raises(ValueError):
        to_rgb565(np.zeros((SIZE, SIZE), dtype=np.uint8))
    with pytest.raises(ValueError):
        to_rgb565(np.zeros((SIZE, SIZE, 4), dtype=np.uint8))


def test_to_rgb565_accepts_noncontiguous_views() -> None:
    frame = _frame(7)
    flipped = np.fliplr(frame)  # negative-stride view, as the app produces
    assert np.array_equal(to_rgb565(flipped), to_rgb565(flipped.copy()))


# --------------------------------------------------------------------------- #
# FramebufferOutput (against tmp regular files; mmap works on those)
# --------------------------------------------------------------------------- #


def test_framebuffer_writes_both_eyes(tmp_path) -> None:
    p_left = tmp_path / "fbL"
    p_right = tmp_path / "fbR"
    for p in (p_left, p_right):
        p.write_bytes(b"\x00" * (SIZE * SIZE * 2))
    left, right = _frame(1), _frame(2)

    out = FramebufferOutput((str(p_left), str(p_right)))
    out.show(left, right)
    out.close()

    assert p_left.read_bytes() == to_rgb565(left).tobytes()
    assert p_right.read_bytes() == to_rgb565(right).tobytes()


def test_framebuffer_stride_aware_top_left_write(tmp_path) -> None:
    # Fake a 480px-wide framebuffer: line_length 960 bytes.
    stride_px = 480
    stride_bytes = stride_px * 2
    p_left = tmp_path / "wideL"
    p_right = tmp_path / "wideR"
    for p in (p_left, p_right):
        p.write_bytes(b"\x00" * (stride_bytes * SIZE))
    left, right = _frame(3), _frame(4)

    out = FramebufferOutput((str(p_left), str(p_right)), stride_px=stride_px)
    out.show(left, right)
    out.close()

    for path, frame in ((p_left, left), (p_right, right)):
        raw = np.frombuffer(path.read_bytes(), dtype=np.uint8)
        rows = raw.reshape(SIZE, stride_bytes)
        expected = to_rgb565(frame).view(np.uint8).reshape(SIZE, SIZE * 2)
        assert np.array_equal(rows[:, : SIZE * 2], expected)
        # Padding beyond the panel width must stay untouched.
        assert not rows[:, SIZE * 2 :].any()


def test_framebuffer_per_device_stride(tmp_path) -> None:
    p_left = tmp_path / "narrow"
    p_right = tmp_path / "wide"
    p_left.write_bytes(b"\x00" * (SIZE * SIZE * 2))
    p_right.write_bytes(b"\x00" * (480 * 2 * SIZE))
    left, right = _frame(5), _frame(6)

    out = FramebufferOutput((str(p_left), str(p_right)), stride_px=(SIZE, 480))
    out.show(left, right)
    out.close()

    assert p_left.read_bytes() == to_rgb565(left).tobytes()
    rows = np.frombuffer(p_right.read_bytes(), dtype=np.uint8).reshape(SIZE, 960)
    expected = to_rgb565(right).view(np.uint8).reshape(SIZE, SIZE * 2)
    assert np.array_equal(rows[:, : SIZE * 2], expected)


def test_framebuffer_repeated_show_overwrites(tmp_path) -> None:
    p_left = tmp_path / "fbL"
    p_right = tmp_path / "fbR"
    for p in (p_left, p_right):
        p.write_bytes(b"\xff" * (SIZE * SIZE * 2))
    out = FramebufferOutput((str(p_left), str(p_right)))
    out.show(_frame(1), _frame(2))
    final_left, final_right = _frame(8), _frame(9)
    out.show(final_left, final_right)
    out.close()
    assert p_left.read_bytes() == to_rgb565(final_left).tobytes()
    assert p_right.read_bytes() == to_rgb565(final_right).tobytes()


def test_framebuffer_too_small_device_raises_clearly(tmp_path) -> None:
    p_left = tmp_path / "tiny"
    p_right = tmp_path / "ok"
    p_left.write_bytes(b"\x00" * 1000)
    p_right.write_bytes(b"\x00" * (SIZE * SIZE * 2))
    with pytest.raises(ValueError, match="too small"):
        FramebufferOutput((str(p_left), str(p_right)))


def test_framebuffer_stride_narrower_than_panel_raises(tmp_path) -> None:
    p = tmp_path / "fb"
    p.write_bytes(b"\x00" * (SIZE * SIZE * 2))
    with pytest.raises(ValueError, match="narrower"):
        FramebufferOutput((str(p), str(p)), stride_px=100)


def test_framebuffer_wrong_frame_shape_raises(tmp_path) -> None:
    p = tmp_path / "fb"
    p.write_bytes(b"\x00" * (SIZE * SIZE * 2))
    out = FramebufferOutput((str(p), str(p)))
    try:
        bad = np.zeros((120, 120, 3), dtype=np.uint8)
        with pytest.raises(ValueError, match="shape"):
            out.show(bad, bad)
    finally:
        out.close()


def test_framebuffer_close_is_idempotent(tmp_path) -> None:
    p = tmp_path / "fb"
    p.write_bytes(b"\x00" * (SIZE * SIZE * 2))
    out = FramebufferOutput((str(p), str(p)))
    out.close()
    out.close()  # must not raise


# --------------------------------------------------------------------------- #
# RecordOutput
# --------------------------------------------------------------------------- #


def test_record_writes_composite_pngs(tmp_path) -> None:
    rec_dir = tmp_path / "rec"
    rec = RecordOutput(rec_dir)
    frames = [(_frame(i), _frame(100 + i)) for i in range(5)]
    for left, right in frames:
        rec.show(left, right)
    rec.close()

    pngs = sorted(rec_dir.glob("*.png"))
    assert [p.name for p in pngs] == [f"frame_{i:05d}.png" for i in range(5)]
    expected_size = (2 * SIZE + RecordOutput.GAP, SIZE)  # (width, height)
    with Image.open(pngs[0]) as im:
        assert im.size == expected_size
        arr = np.asarray(im.convert("RGB"))
    left0, right0 = frames[0]
    assert np.array_equal(arr[:, :SIZE], left0)
    assert np.array_equal(arr[:, SIZE + RecordOutput.GAP :], right0)
    assert not arr[:, SIZE : SIZE + RecordOutput.GAP].any()  # black gap


def test_record_writes_gif_on_close(tmp_path) -> None:
    gif = tmp_path / "anim.gif"
    rec = RecordOutput(tmp_path / "rec", gif_path=gif)
    for i in range(5):
        rec.show(_frame(i), _frame(50 + i))
    rec.close()
    assert gif.exists()
    with Image.open(gif) as im:
        assert im.format == "GIF"
        assert im.n_frames == 5
        assert im.size == (2 * SIZE + RecordOutput.GAP, SIZE)


def test_record_no_gif_when_not_requested(tmp_path) -> None:
    rec = RecordOutput(tmp_path / "rec")
    rec.show(_frame(0), _frame(1))
    rec.close()
    assert list(tmp_path.glob("*.gif")) == []


def test_record_every_and_max_frames(tmp_path) -> None:
    rec_dir = tmp_path / "rec"
    rec = RecordOutput(rec_dir, max_frames=2, every=2)
    for i in range(6):
        rec.show(_frame(i), _frame(i))  # saves shows 0 and 2, then stops
    rec.close()
    pngs = sorted(rec_dir.glob("*.png"))
    assert [p.name for p in pngs] == ["frame_00000.png", "frame_00001.png"]


def test_record_close_idempotent_and_empty_ok(tmp_path) -> None:
    rec = RecordOutput(tmp_path / "rec", gif_path=tmp_path / "anim.gif")
    rec.close()
    rec.close()
    assert not (tmp_path / "anim.gif").exists()  # no frames -> no gif


# --------------------------------------------------------------------------- #
# PreviewOutput (headless via SDL dummy driver)
# --------------------------------------------------------------------------- #


def test_preview_pushes_quit_on_window_close_and_esc(monkeypatch) -> None:
    pygame = pytest.importorskip("pygame")
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    from spookyeyes.outputs.preview import PreviewOutput

    events: queue.Queue = queue.Queue()
    pv = PreviewOutput(scale=1, events=events)
    try:
        left, right = _frame(1), _frame(2)
        pv.show(left, right)
        assert events.empty()

        pygame.event.post(pygame.event.Event(pygame.QUIT))
        pv.show(left, right)
        assert events.get_nowait().kind == "quit"

        pygame.event.post(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_ESCAPE))
        pv.show(left, right)
        assert events.get_nowait().kind == "quit"

        pv.close()
        pv.close()  # idempotent
        pv.show(left, right)  # after close: no-op, must not raise
    finally:
        pv.close()


# --------------------------------------------------------------------------- #
# make_output factory
# --------------------------------------------------------------------------- #


def test_make_output_null_and_protocol() -> None:
    out = make_output(DisplayConfig(output="null"))
    assert isinstance(out, NullOutput)
    assert isinstance(out, Output)  # runtime-checkable protocol
    out.show(_frame(0), _frame(1))
    out.close()


def test_make_output_fb_uses_config_paths(tmp_path) -> None:
    p_left = tmp_path / "fb1"
    p_right = tmp_path / "fb2"
    for p in (p_left, p_right):
        p.write_bytes(b"\x00" * (SIZE * SIZE * 2))
    cfg = DisplayConfig(output="fb", fb_left=str(p_left), fb_right=str(p_right))
    out = make_output(cfg)
    assert isinstance(out, FramebufferOutput)
    left = _frame(3)
    out.show(left, _frame(4))
    out.close()
    assert p_left.read_bytes() == to_rgb565(left).tobytes()


def test_make_output_record_uses_config_dir_and_gif(tmp_path) -> None:
    cfg = DisplayConfig(output="record", record_dir=str(tmp_path / "rec"))
    out = make_output(cfg, gif_path=str(tmp_path / "a.gif"))
    assert isinstance(out, RecordOutput)
    out.show(_frame(0), _frame(1))
    out.close()
    assert (tmp_path / "rec" / "frame_00000.png").exists()
    assert (tmp_path / "a.gif").exists()


def test_make_output_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match="unknown"):
        make_output(DisplayConfig(output="hologram"))
