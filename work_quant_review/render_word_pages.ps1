param(
    [Parameter(Mandatory = $true)][string]$DocxPath,
    [Parameter(Mandatory = $true)][string]$OutputDir,
    [int]$StartPage = 1,
    [int]$EndPage = 0
)

$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
$captureSource = @'
using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Runtime.InteropServices;

public static class WordWindowCapture
{
    [StructLayout(LayoutKind.Sequential)]
    public struct RECT { public int Left; public int Top; public int Right; public int Bottom; }

    [DllImport("user32.dll")]
    private static extern bool GetWindowRect(IntPtr hWnd, out RECT rect);

    [DllImport("user32.dll")]
    private static extern bool PrintWindow(IntPtr hWnd, IntPtr hdcBlt, uint flags);

    public static void Save(IntPtr hwnd, string path)
    {
        RECT rect;
        if (!GetWindowRect(hwnd, out rect)) throw new InvalidOperationException("GetWindowRect failed.");
        int width = Math.Max(1, rect.Right - rect.Left);
        int height = Math.Max(1, rect.Bottom - rect.Top);
        using (var bitmap = new Bitmap(width, height, PixelFormat.Format32bppArgb))
        using (var graphics = Graphics.FromImage(bitmap))
        {
            IntPtr hdc = graphics.GetHdc();
            try
            {
                if (!PrintWindow(hwnd, hdc, 2)) throw new InvalidOperationException("PrintWindow failed.");
            }
            finally { graphics.ReleaseHdc(hdc); }
            bitmap.Save(path, ImageFormat.Png);
        }
    }
}
'@
Add-Type -TypeDefinition $captureSource -ReferencedAssemblies 'System.Drawing.dll'

$docx = (Resolve-Path -LiteralPath $DocxPath).Path
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
$output = (Resolve-Path -LiteralPath $OutputDir).Path

$word = New-Object -ComObject Word.Application
$word.Visible = $true
$word.DisplayAlerts = 0
$word.ScreenUpdating = $true
$word.Options.CheckSpellingAsYouType = $false
$word.Options.CheckGrammarAsYouType = $false

try {
    $doc = $word.Documents.Open($docx, $false, $true, $false)
    $window = $doc.ActiveWindow
    $window.WindowState = 1
    $window.View.Type = 3
    $pages = $window.Panes.Item(1).Pages
    $pageCount = $pages.Count
    $pageStart = [Math]::Max(1, $StartPage)
    $pageEnd = if ($EndPage -gt 0) { [Math]::Min($pageCount, $EndPage) } else { $pageCount }

    for ($page = $pageStart; $page -le $pageEnd; $page++) {
        $target = Join-Path $output ("page-{0:D2}.png" -f $page)
        $bits = $pages.Item($page).EnhMetaFileBits
        $stream = [IO.MemoryStream]::new($bits)
        $metafile = [Drawing.Image]::FromStream($stream)
        $bitmap = [Drawing.Bitmap]::new($metafile.Width * 2, $metafile.Height * 2)
        $graphics = [Drawing.Graphics]::FromImage($bitmap)
        try {
            $graphics.Clear([Drawing.Color]::White)
            $graphics.InterpolationMode = [Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
            $graphics.SmoothingMode = [Drawing.Drawing2D.SmoothingMode]::HighQuality
            $graphics.DrawImage($metafile, 0, 0, $bitmap.Width, $bitmap.Height)
            $bitmap.Save($target, [Drawing.Imaging.ImageFormat]::Png)
        }
        finally {
            $graphics.Dispose()
            $bitmap.Dispose()
            $metafile.Dispose()
            $stream.Dispose()
        }
    }

    $doc.Close(0)
    Write-Output ("PAGES_CAPTURED={0}" -f $pageCount)
}
finally {
    $word.Quit()
}
