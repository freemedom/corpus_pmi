<#
.SYNOPSIS
按正则找出文本文件中的命中行，默认输出前 5 行。

.DESCRIPTION
忽略大小写和重音，规则与 query_blob.py 的 fold 相同：
œ 收成 oe，æ 收成 ae，去掉组合重音。直撇号与弯撇号仍然不同。
正则的转义和量词保持原义，只折叠字面字符。œ+ 会折成 (?:oe)+。
方括号里的 œ 会变成 o 和 e 两个字符。
命中行按原文输出。-Count 默认 5，0 表示不截断。

.EXAMPLE
.\Select-MatchingLines.ps1 -Path .\monde.txt -Pattern "partout dans"

.EXAMPLE
.\Select-MatchingLines.ps1 .\monde.txt "ete" -Count 0
#>
param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string] $Path,

    [Parameter(Mandatory = $true, Position = 1)]
    [AllowEmptyString()]
    [string] $Pattern,

    [Parameter(Position = 2)]
    [int] $Count = 5
)

$ErrorActionPreference = "Stop"

if (-not ("FrenchTcf.Collocate.TextFold" -as [type])) {
    $source = Join-Path $PSScriptRoot "Select-MatchingLines.cs"
    if (-not (Test-Path -LiteralPath $source)) { # -LiteralPath 表示这段字符串就是文件路径本身，PowerShell 不会把它当成通配符。 -Path 会把 *、? 和方括号当成通配符。
        throw "找不到折叠代码: $source"
    }
    Add-Type -Path $source
}

if ([string]::IsNullOrEmpty($Pattern)) {
    throw "Pattern 不能为空"
}
if ($Count -lt 0) {
    throw "Count 不能小于 0"
}
if (-not (Test-Path -LiteralPath $Path)) {
    throw "找不到文件: $Path"
}
$item = Get-Item -LiteralPath $Path
if ($item.PSIsContainer) {
    throw "不是文件: $Path"
}

$foldedPattern = [FrenchTcf.Collocate.TextFold]::FoldPattern($Pattern)
if ([string]::IsNullOrEmpty($foldedPattern)) {
    throw "Pattern 折叠后为空: $Pattern"
}

$regexOptions = [System.Text.RegularExpressions.RegexOptions]::CultureInvariant
try {
    $regex = New-Object System.Text.RegularExpressions.Regex($foldedPattern, $regexOptions)
}
catch {
    $detail = $_.Exception.Message
    if ($_.Exception.InnerException) {
        $detail = $_.Exception.InnerException.Message
    }
    throw ("非法正则: {0}`n折叠后: {1}`n{2}" -f $Pattern, $foldedPattern, $detail)
}

$result = [FrenchTcf.Collocate.TextFold]::Search($item.FullName, $regex, $Count)
foreach ($line in $result.Lines) {
    Write-Output $line
}
Write-Host ("命中 {0} 条，输出 {1} 条" -f $result.Matched, $result.Lines.Length)
