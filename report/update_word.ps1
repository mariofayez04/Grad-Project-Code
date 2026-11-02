# Lets Word refresh the contents pages and lists, saves the report, and exports a PDF for checking.
$docx = Join-Path $PSScriptRoot "Report2 Mario Fayez Sobhy 23510186.docx"
$pdf = [System.IO.Path]::ChangeExtension($docx, ".pdf")
$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($docx)
    foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
    foreach ($list in $doc.TablesOfFigures) { $list.Update() }
    $doc.Fields.Update() | Out-Null
    $doc.Save()
    $doc.ExportAsFixedFormat($pdf, 17)   # 17 = PDF
    "Pages: $($doc.ComputeStatistics(2)); words: $($doc.ComputeStatistics(0))"
    $doc.Close(0)
} finally {
    $word.Quit()
}
