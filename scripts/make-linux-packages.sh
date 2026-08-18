#!/usr/bin/env bash
# Build a .deb and an AppImage from an already-built Linux release.
#
# Both are "install it the way this platform installs things", which the tar.gz
# is not: a tarball asks the user to decide where binaries go and to edit their
# PATH, and neither answer is obvious to somebody who just wants to run a model.
#
# **Neither needs a packaging toolchain.** A .deb is an `ar` archive of three
# members, and an AppImage is a squashfs image with a runtime prepended -- both
# are built here from coreutils plus whatever is already on a GitHub runner, so
# there is no dpkg-dev, no fpm, no linuxdeploy, and nothing that can drift.
#
#   scripts/make-linux-packages.sh <version> <staging-dir>
#
# Writes `chaos_<version>_amd64.deb` and `Chaos-<version>-x86_64.AppImage` into
# the working directory.
set -euo pipefail

VER="${1:?usage: make-linux-packages.sh <version> <staging-dir>}"
SRC="${2:?usage: make-linux-packages.sh <version> <staging-dir>}"
# Debian versions may not start with a letter, and tags here are `v0.0.7`.
DEBVER="${VER#v}"
ARCH=amd64

BINS="chaos-app chaos-run chaos-serve chaos-probe chaos-model-info chaos-pull chaos-meta gguf-info chaos-loadbench chaos-iobench chaos-gpubench"

# `chaos-app` is the Windows window; on Linux it is a stub that prints a line
# and exits, so it is packaged but never made the desktop entry's Exec.
DESKTOP_EXEC=chaos-run

say() { printf '  %s\n' "$*"; }

# -- the .deb ---------------------------------------------------------------
#
# ar archive, members in this exact order: debian-binary, control.tar.gz,
# data.tar.gz. dpkg reads them positionally and rejects any other order.
build_deb() {
  local root=deb-root
  rm -rf "$root" && mkdir -p "$root/usr/lib/chaos" "$root/usr/bin" "$root/DEBIAN" \
    "$root/usr/share/doc/chaos" "$root/usr/share/applications"

  for b in $BINS; do
    [ -f "$SRC/$b" ] || { say "missing $b"; return 1; }
    install -m 755 "$SRC/$b" "$root/usr/lib/chaos/$b"
    # Symlinks rather than copies: eleven binaries in /usr/bin is eleven things
    # to collide with, and /usr/lib/<pkg> is where Debian puts a private set.
    ln -sf "/usr/lib/chaos/$b" "$root/usr/bin/$b"
  done
  install -m 644 README.md LICENSE NOTICE "$root/usr/share/doc/chaos/" 2>/dev/null || true

  local size
  size=$(du -ks "$root/usr" | cut -f1)

  cat > "$root/DEBIAN/control" <<EOF
Package: chaos
Version: $DEBVER
Section: science
Priority: optional
Architecture: $ARCH
Maintainer: Atur Dana <https://github.com/aturzone>
Installed-Size: $size
Homepage: https://github.com/aturzone/Chaos
Description: Run language models larger than your memory
 Chaos keeps the always-read weights resident and streams routed experts from
 disk per token, so a Mixture-of-Experts container far larger than RAM still
 generates text. A 155 GB model runs on a 16 GB machine.
 .
 Includes the runner, an OpenAI-compatible server, the downloader and the
 measurement tools.
EOF

  cat > "$root/usr/share/applications/chaos.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Chaos
Comment=Run language models larger than your memory
Exec=$DESKTOP_EXEC
Terminal=true
Categories=Development;Science;
EOF

  # md5sums, which `debsums` and some tooling expect. Paths are relative to /.
  ( cd "$root" && find usr -type f -exec md5sum {} + > DEBIAN/md5sums )

  ( cd "$root" && tar czf ../control.tar.gz -C DEBIAN . )
  ( cd "$root" && tar czf ../data.tar.gz usr )
  echo "2.0" > debian-binary

  local out="chaos_${DEBVER}_${ARCH}.deb"
  rm -f "$out"
  ar rc "$out" debian-binary control.tar.gz data.tar.gz
  rm -f debian-binary control.tar.gz data.tar.gz
  rm -rf "$root"
  say "wrote $out ($(du -h "$out" | cut -f1))"
}

# -- the AppImage -----------------------------------------------------------
#
# An AppDir plus the upstream runtime concatenated with a squashfs image. The
# runtime is a small ELF that mounts the image and executes AppRun; it is
# downloaded because it is a prebuilt binary from the AppImage project and there
# is nothing to build.
build_appimage() {
  local dir=Chaos.AppDir
  rm -rf "$dir" && mkdir -p "$dir/usr/bin"
  for b in $BINS; do
    install -m 755 "$SRC/$b" "$dir/usr/bin/$b"
  done

  cat > "$dir/AppRun" <<'EOF'
#!/bin/sh
# Any binary in the image, by name: `./Chaos.AppImage chaos-probe --quick`.
# With no argument it runs the runner, which is what the desktop entry does.
HERE="$(dirname "$(readlink -f "$0")")"
export PATH="$HERE/usr/bin:$PATH"
if [ $# -gt 0 ] && [ -x "$HERE/usr/bin/$1" ]; then
  prog="$1"; shift
  exec "$HERE/usr/bin/$prog" "$@"
fi
exec "$HERE/usr/bin/chaos-run" "$@"
EOF
  chmod 755 "$dir/AppRun"

  cat > "$dir/chaos.desktop" <<EOF
[Desktop Entry]
Type=Application
Name=Chaos
Comment=Run language models larger than your memory
Exec=AppRun
Icon=chaos
Terminal=true
Categories=Development;Science;
EOF
  # The spec wants an icon at the AppDir root; assets/logo.png is generated from
  # the same SVG as everything else.
  if [ -f assets/logo.png ]; then
    cp assets/logo.png "$dir/chaos.png"
  fi

  local runtime=runtime-x86_64
  if ! curl -fsSL -o "$runtime" \
      "https://github.com/AppImage/type2-runtime/releases/download/continuous/runtime-x86_64"; then
    say "could not fetch the AppImage runtime; skipping the AppImage"
    rm -rf "$dir"
    return 0
  fi
  chmod +x "$runtime"

  local out="Chaos-${VER}-x86_64.AppImage"
  rm -f "$out" chaos.squashfs
  # -root-owned so the image does not carry the runner's uid; -noappend so a
  # rerun does not silently add a second copy of everything.
  mksquashfs "$dir" chaos.squashfs -root-owned -noappend -quiet -comp zstd
  cat "$runtime" chaos.squashfs > "$out"
  chmod +x "$out"
  rm -f chaos.squashfs "$runtime"
  rm -rf "$dir"
  say "wrote $out ($(du -h "$out" | cut -f1))"
}

say "packaging Chaos $VER from $SRC"
build_deb
if command -v mksquashfs >/dev/null 2>&1; then
  build_appimage
else
  say "mksquashfs is not installed; skipping the AppImage"
fi
