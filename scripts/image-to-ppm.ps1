# Convert any image Windows can open into a binary PPM (P6).
#
# **Why a converter rather than a decoder in the crate.** The autoencoder's
# round-trip needs a real photograph on the input side, and `chaos-image` has a
# PNG *encoder* only -- reading one back would mean an inflate implementation
# whose bugs would be charged to the model. P6 is a nine-byte ASCII header and
# then RGB bytes, so the Rust side parses it in twenty lines that cannot be
# subtly wrong.
#
# GDI+ does the decoding, which is the same thing that verified the PNG encoder's
# pixel values.
#
#   powershell -File scripts/image-to-ppm.ps1 -In photo.jpg -Out photo.ppm -Size 256
#
# The image is centre-cropped to a square first, so the aspect ratio survives and
# the result is a multiple of 8 in both directions -- which the encoder requires,
# since three stride-2 convolutions cannot halve an odd number evenly.

param(
    [Parameter(Mandatory = $true)][string]$In,
    [Parameter(Mandatory = $true)][string]$Out,
    [int]$Size = 256
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing

if ($Size % 8 -ne 0) { throw "size $Size is not a multiple of 8" }

$src = [System.Drawing.Image]::FromFile((Resolve-Path $In).Path)
try {
    # Centre crop to a square, then scale. Cropping before scaling keeps the
    # subject's proportions; scaling a non-square image to a square would stretch
    # it, and a stretched photo is still a fair test but a confusing one to look at.
    $side = [Math]::Min($src.Width, $src.Height)
    $sx = [int](($src.Width - $side) / 2)
    $sy = [int](($src.Height - $side) / 2)

    $bmp = New-Object System.Drawing.Bitmap($Size, $Size, [System.Drawing.Imaging.PixelFormat]::Format24bppRgb)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    try {
        $g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
        $g.PixelOffsetMode = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality
        $g.DrawImage($src,
            (New-Object System.Drawing.Rectangle(0, 0, $Size, $Size)),
            (New-Object System.Drawing.Rectangle($sx, $sy, $side, $side)),
            [System.Drawing.GraphicsUnit]::Pixel)
    } finally { $g.Dispose() }

    # LockBits rather than GetPixel: GetPixel on 256x256 is 65,536 marshalled
    # calls and takes seconds.
    $rect = New-Object System.Drawing.Rectangle(0, 0, $Size, $Size)
    $data = $bmp.LockBits($rect, [System.Drawing.Imaging.ImageLockMode]::ReadOnly,
        [System.Drawing.Imaging.PixelFormat]::Format24bppRgb)
    try {
        $raw = New-Object byte[] ($data.Stride * $Size)
        [System.Runtime.InteropServices.Marshal]::Copy($data.Scan0, $raw, 0, $raw.Length)

        $rgb = New-Object byte[] ($Size * $Size * 3)
        for ($y = 0; $y -lt $Size; $y++) {
            $row = $y * $data.Stride
            $dst = $y * $Size * 3
            for ($x = 0; $x -lt $Size; $x++) {
                # GDI+ 24bpp is B, G, R in memory. PPM is R, G, B.
                $rgb[$dst + $x * 3 + 0] = $raw[$row + $x * 3 + 2]
                $rgb[$dst + $x * 3 + 1] = $raw[$row + $x * 3 + 1]
                $rgb[$dst + $x * 3 + 2] = $raw[$row + $x * 3 + 0]
            }
        }
    } finally { $bmp.UnlockBits($data) }

    $header = [System.Text.Encoding]::ASCII.GetBytes("P6`n$Size $Size`n255`n")
    $stream = [System.IO.File]::Create((New-Item -ItemType File -Path $Out -Force).FullName)
    try {
        $stream.Write($header, 0, $header.Length)
        $stream.Write($rgb, 0, $rgb.Length)
    } finally { $stream.Dispose() }

    Write-Host "wrote $Out -- ${Size}x${Size}, $($rgb.Length) bytes of RGB"
} finally {
    $src.Dispose()
    if ($bmp) { $bmp.Dispose() }
}
