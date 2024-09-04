#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Script to build asymptote
.DESCRIPTION
    Builds asymptote installer file
.PARAMETER Version
    Specifies Asymptote version to build.
#>
param(
    [Parameter(Mandatory)]
    [string]$Version
)

# ----------------------------------------------------
# checking documentation files
$extfilesRoot="Z:\asy\asydoc"
$requiredDocumentationFiles=@(
    "asymptote.sty"
    "asymptote.pdf"
    "asy-latex.pdf"
    "CAD.pdf"
    "TeXShopAndAsymptote.pdf"
    "asyRefCard.pdf"
    "latexusage.pdf"
)

$hasDocFiles=$true
foreach ($requiredDocFile in $requiredDocumentationFiles) {
    if (-Not (Test-Path -PathType leaf "$extfilesRoot/$requiredDocFile")) {
        $hasDocFiles=$false
        Write-Error "$requiredDocFile not found.
Please ensure $requiredDocFile is available in $extfilesRoot directory"
    }
}

if (-Not $hasDocFiles) {
    Write-Error "Documentation file(s) are not present in $extFilesRoot directory. Will not build asymptote."
    Break
}

# ----------------------------------------------------
# tools cache
$toolscacheRoot="tools-cache"
New-Item -ItemType Directory -Path $toolscacheRoot -Force

$useToolsCacheVcpkg=$false

# tools cache variables
$vcpkgSha256="e590c2b30c08caf1dd8d612ec602a003f9784b7d"

# vcpkg
if (-Not $env:VCPKG_ROOT)
{
    $vcpkgToolsCacheLoc = "$toolscacheRoot/vcpkg"
    Write-Host "VCPKG_ROOT Not found, checking for $vcpkgToolsCacheLoc"
    if (-Not (Test-Path -PathType Container $vcpkgToolsCacheLoc))
    {
        git clone https://github.com/microsoft/vcpkg.git "$vcpkgToolsCacheLoc"
    }
    else
    {
        Write-Host "vcpkg directory found"
    }
    git -C $vcpkgToolsCacheLoc fetch
    git -C $vcpkgToolsCacheLoc reset --hard $vcpkgSha256

    if (-Not (Test-Path $vcpkgToolsCacheLoc/vcpkg.exe))
    {
        Push-Location $vcpkgToolsCacheLoc
        & ./bootstrap-vcpkg.bat
        Pop-Location
    }

    $useToolsCacheVcpkg=true
}
else
{
    Write-Host "Using VCPKG_ROOT at $($env:VCPKG_ROOT)"
}

# checking for NSIS
$baseNsisExec=Get-Command makensis -ErrorAction ignore
if ($null -eq $baseNsisExec)
{
    Write-Host "makensis not found. Checking for downloaded NSIS."
    $makeNsisLoc="$toolscacheRoot/nsis/makensis.exe"

    if (-Not (Test-Path -PathType leaf $makeNsisLoc))
    {
        Write-Error "$makeNsisLoc not found. Please ensure you download and extract NSIS Zip to $toolscacheRoot/nsis,
found at https://sourceforge.net/projects/nsis/
"
        Break
    }
}
else
{
    $makeNsisLoc=$baseNsisExec.Path
}

# python
$pyVenvLocation="$toolscacheRoot/pyxasy"
$pyXasyActivateScript="$pyVenvLocation/Scripts/activate.ps1"
if (-Not (Test-Path -PathType leaf $pyXasyActivateScript))
{
    python -m virtualenv $pyVenvLocation
}

# ----------------------------------------------------
# cloning asymptote
if (Test-Path asymptote)
{
    Remove-Item -Force -Recurse asymptote
}
# TODO: Once this is merged into master, the "-b msvc-suppoort-make" argument which
#       tells git to checkout to msvc-support-make branch can be removed
git clone --depth=1 -b msvc-support-make https://github.com/vectorgraphics/asymptote
Copy-Item -Recurse -Force -Path "$extfilesRoot" -Destination "asymptote/extfiles"


# ----------------------------------------------------
# build GUI
& $pyXasyActivateScript
Push-Location asymptote/GUI
& python -m pip install -r requirements.txt -r requirements.dev.txt
& python buildtool.py build --version-override=$Version
Pop-Location


# ----------------------------------------------------
# build C++ side
Import-VisualStudioVars -Architecture x64
Push-EnvironmentBlock
    $env:ASY_VERSION_OVERRIDE = $Version
    if ($useToolsCacheVcpkg)
    {
        $env:VCPKG_ROOT = $vcpkgToolsCacheLoc
    }
    Push-Location asymptote
    cmake --preset msvc/release-with-external-doc-files
    Pop-Location

    $asymptoteCmakeConfigurationDir="asymptote/cmake-build-msvc/release"
    cmake --build $asymptoteCmakeConfigurationDir --target asy-pre-nsis-targets -j
    # install to pre-installation root
    cmake --install $asymptoteCmakeConfigurationDir --component asy-pre-nsis

Pop-EnvironmentBlock  # ASY_VERSION_OVERRIDE, VCPKG_ROOT
Pop-EnvironmentBlock  # Visual studio vars

# ------------------------------------------------------
# Generate NSIS installer file
& ./asymptote/cmake-install-w32-nsis-release/build-asy-installer.ps1 "$makeNsisLoc"

# TODO: Also handle CTAN builds
