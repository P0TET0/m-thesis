$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.IO.Compression
Add-Type -AssemblyName System.IO.Compression.FileSystem

$repoRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '../..')).Path
$bundleName = 'chat_handoff_exp007_20261021'
$bundleDir = Join-Path $PSScriptRoot ('exports/' + $bundleName)
$archivePath = Join-Path $PSScriptRoot ('exports/' + $bundleName + '.zip')
if (Test-Path -LiteralPath $archivePath) { throw 'Archive already exists; refusing to overwrite.' }

$entries = [System.Collections.Generic.List[object]]::new()
function Add-BundleFile([string]$source, [string]$relative) {
    $item = Get-Item -LiteralPath $source
    $entries.Add([pscustomobject]@{
        source = $item.FullName
        path = $relative.Replace('\', '/')
        bytes = $item.Length
        sha256 = (Get-FileHash -LiteralPath $item.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    })
}
foreach ($name in @('00_README_ja.txt', '01_context_summary_ja.txt', '02_report_to_supervisor_ja.txt')) {
    Add-BundleFile (Join-Path $bundleDir $name) $name
}

$evidence = @(
    Get-ChildItem -LiteralPath $PSScriptRoot -File | Where-Object { $_.Extension -in @('.py', '.mjs', '.md', '.ps1') }
    Get-ChildItem -LiteralPath (Join-Path $PSScriptRoot 'reports') -File | Where-Object { $_.Extension -in @('.md', '.txt') }
    Get-ChildItem -LiteralPath (Join-Path $PSScriptRoot 'data/processed') -File | Where-Object { $_.Extension -in @('.csv', '.json') }
    Get-ChildItem -LiteralPath (Join-Path $PSScriptRoot 'figures') -Recurse -File | Where-Object { $_.Extension -in @('.png', '.pdf') }
)
foreach ($item in ($evidence | Sort-Object FullName)) {
    Add-BundleFile $item.FullName ([System.IO.Path]::GetRelativePath($repoRoot, $item.FullName))
}
foreach ($relative in @(
    'experiments/exp006/reports/focus_broad_families_uniform_axes/focus_broad_families_y_equals_x_metrics_report.md',
    'experiments/exp006/data/processed/focus_broad_families_uniform_axes/focus_broad_families_y_equals_x_metrics.csv',
    'experiments/exp006/data/processed/focus_broad_families_uniform_axes/focus_broad_families_y_equals_x_ranking_all.csv'
)) { Add-BundleFile (Join-Path $repoRoot $relative) $relative }
Add-BundleFile (Join-Path $repoRoot 'data/output/chat_handoff_exp005_exp006_summary.zip') 'background/chat_handoff_exp005_exp006_summary.zip'

$distinctCount = @($entries.path | Sort-Object -Unique).Count
if ($distinctCount -ne $entries.Count) { throw 'Duplicate archive names.' }
$manifestPath = Join-Path $bundleDir 'file_manifest.csv'
$entries | Select-Object path, bytes, sha256 | Export-Csv -LiteralPath $manifestPath -NoTypeInformation -Encoding utf8BOM
$utf8 = [System.Text.UTF8Encoding]::new($false)
$archive = [System.IO.Compression.ZipFile]::Open($archivePath, [System.IO.Compression.ZipArchiveMode]::Create)
try {
    foreach ($record in $entries) {
        [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($archive, $record.source, ($bundleName + '/' + $record.path), [System.IO.Compression.CompressionLevel]::Optimal) | Out-Null
    }
    [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($archive, $manifestPath, ($bundleName + '/file_manifest.csv')) | Out-Null
} finally { $archive.Dispose() }

$archive = [System.IO.Compression.ZipFile]::OpenRead($archivePath)
try {
    if ($archive.Entries.Count -ne ($entries.Count + 1)) { throw 'Archive entry count mismatch.' }
    foreach ($record in $entries) {
        $entry = $archive.GetEntry($bundleName + '/' + $record.path)
        if ($null -eq $entry -or $entry.Length -ne $record.bytes) { throw ('Size mismatch: ' + $record.path) }
        $stream = $entry.Open()
        $hasher = [System.Security.Cryptography.SHA256]::Create()
        try { $hash = [Convert]::ToHexString($hasher.ComputeHash($stream)).ToLowerInvariant() }
        finally { $stream.Dispose(); $hasher.Dispose() }
        if ($hash -ne $record.sha256) { throw ('Archive hash mismatch: ' + $record.path) }
        if ((Get-FileHash -LiteralPath $record.source -Algorithm SHA256).Hash.ToLowerInvariant() -ne $record.sha256) {
            throw ('Source changed during packaging: ' + $record.path)
        }
    }
} finally { $archive.Dispose() }

$validationPath = Join-Path $bundleDir 'package_validation.txt'
$rawBytes = ($entries | Measure-Object -Property bytes -Sum).Sum
$validationText = @(
    'Packaging validation: PASS',
    ('Checked payload files: ' + $entries.Count),
    ('Payload bytes before compression: ' + $rawBytes),
    'Every payload entry was read back from the ZIP and matched to its SHA-256 and byte count.',
    'All input file hashes were rechecked after packaging; no input changes detected.',
    'file_manifest.csv covers all payload files, excluding the manifest and this validation log.',
    'This is a summarized research handoff, not a verbatim chat transcript or a self-contained execution environment.'
) -join "`r`n"
[System.IO.File]::WriteAllText($validationPath, $validationText, $utf8)
$archive = [System.IO.Compression.ZipFile]::Open($archivePath, [System.IO.Compression.ZipArchiveMode]::Update)
try { [System.IO.Compression.ZipFileExtensions]::CreateEntryFromFile($archive, $validationPath, ($bundleName + '/package_validation.txt')) | Out-Null }
finally { $archive.Dispose() }

$archive = [System.IO.Compression.ZipFile]::OpenRead($archivePath)
try {
    if ($archive.Entries.Count -ne ($entries.Count + 2)) { throw 'Final entry count mismatch.' }
    foreach ($name in @('file_manifest.csv', 'package_validation.txt')) {
        $entry = $archive.GetEntry($bundleName + '/' + $name)
        $reader = [System.IO.StreamReader]::new($entry.Open(), [System.Text.Encoding]::UTF8, $true)
        try { $actual = $reader.ReadToEnd() } finally { $reader.Dispose() }
        $expected = [System.IO.File]::ReadAllText((Join-Path $bundleDir $name), [System.Text.Encoding]::UTF8)
        if ($actual -ne $expected) { throw ('Metadata mismatch: ' + $name) }
    }
} finally { $archive.Dispose() }
Get-Item -LiteralPath $archivePath | Select-Object FullName, Length
Write-Output ('Payload files: ' + $entries.Count + '; validation PASS')
