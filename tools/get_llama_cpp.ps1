param([string]$Dest)
# Downloads the newest official llama.cpp Windows CUDA x64 build (build releases are tagged bNNNN;
# the "latest" release is only a version marker) plus the matching CUDA runtime DLLs.
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'
New-Item -ItemType Directory -Force -Path $Dest | Out-Null
if (Test-Path (Join-Path $Dest 'llama-server.exe')) { Write-Output "llama-server already present"; exit 0 }
$releases = Invoke-RestMethod -Uri 'https://api.github.com/repos/ggml-org/llama.cpp/releases?per_page=15' -Headers @{ 'User-Agent' = 'hams-setup' }
$bin = $null; $rt = $null; $tag = $null
foreach ($rel in $releases) {
  foreach ($ver in @('cuda-12', 'cuda-13')) {   # CUDA 12.x build first: works with any driver >= 551
    $b = $rel.assets | Where-Object { $_.name -match "^llama-.*-bin-win-$ver.*-x64\.zip$" } | Select-Object -First 1
    $r = $rel.assets | Where-Object { $_.name -match "^cudart-llama-bin-win-$ver.*-x64\.zip$" } | Select-Object -First 1
    if ($b -and $r) { $bin = $b; $rt = $r; $tag = $rel.tag_name; break }
  }
  if ($bin) { break }
}
if (-not $bin) { throw "no CUDA windows asset found in recent releases" }
Write-Output ("release " + $tag + " -> " + $bin.name + " + " + $rt.name)
foreach ($a in @($bin, $rt)) {
  $zip = Join-Path $env:TEMP $a.name
  Write-Output ("download " + $a.name + " (" + [math]::Round($a.size / 1MB) + " MB)")
  Invoke-WebRequest -Uri $a.browser_download_url -OutFile $zip
  Expand-Archive -Path $zip -DestinationPath $Dest -Force
  Remove-Item $zip
}
$exe = Get-ChildItem -Path $Dest -Recurse -Filter 'llama-server.exe' | Select-Object -First 1
if ($exe -and $exe.DirectoryName -ne (Resolve-Path $Dest).Path) {
  Get-ChildItem -Path $exe.DirectoryName | Move-Item -Destination $Dest -Force
}
Set-Content -Path (Join-Path $Dest 'VERSION.txt') -Value ($tag + " " + $bin.name)
& (Join-Path $Dest 'llama-server.exe') --version 2>&1 | Write-Output
Write-Output "llama.cpp ready"
