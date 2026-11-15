# Soft Orbs

A soft, dreamy third-person orb-collecting game made with Rust and Macroquad.

> **About this project:** The game was AI-generated. I made it just to see an example running on my hardware—and to find out what this old laptop can handle.

## Gameplay video
[Watch the gameplay video](soft_orbs.mkv)

## How to play

Collect **20 glowing orbs** while avoiding the gloomlings. Pink hearts restore health, and a shockwave can stun nearby enemies.

| Action | Controls |
| --- | --- |
| Move | WASD |
| Jump / double jump | Space |
| Sprint | Shift |
| Orbit the camera | Mouse, arrow keys, or Q / E |
| Zoom | Mouse wheel |
| Shockwave | F or left-click |
| Pause / resume | Esc or P |
| Quit while paused | Q |

## Run it

With Rust and Cargo installed, run:

```sh
cargo run --release
```

## My hardware

This is the machine I made it to test on:

- **OS:** Debian GNU/Linux x86_64
- **Laptop:** EasyNote MH45 (LX.B190X.005)
- **Window manager:** dwm on X11
- **Kernel:** Linux 6.12.94+deb13-amd64
- **CPU:** Pentium(R) Dual-Core CPU T4200 @ 2.00 GHz (2 cores)
- **GPU:** Intel Mobile 4 Series Chipset Integrated Graphics Controller, 64 MB VRAM
- **Wi-Fi:** Ralink RT2790 Wireless 802.11n 1T/2R PCIe

### Hardware verdict

By modern gaming standards, this hardware is pretty rough: the CPU is an old dual-core, and the integrated graphics has only 64 MB of dedicated VRAM. So, yes—my hardware kind of sucks for modern 3D games. That’s exactly why I wanted to see how this example would run on it. Performance may be limited, and reducing the render scale or turning off post-processing effects may help.

The Wi-Fi adapter is listed for completeness; it doesn’t meaningfully affect the game’s graphics performance.

## Credits and disclosure

The game code was **generated with AI**. I ran it as an example and tested it on the hardware listed above.
