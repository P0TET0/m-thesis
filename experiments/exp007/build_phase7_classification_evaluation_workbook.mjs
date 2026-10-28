import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";


const ROOT = process.cwd();
const EXP007 = path.join(ROOT, "experiments", "exp007");
const PAYLOAD_PATH = path.join(EXP007, "reports", ".phase7_workbook_payload.json");
const OUTPUT_PATH = path.join(EXP007, "reports", "phase7_classification_evaluation.xlsx");
const VALIDATION_LOG = path.join(EXP007, "reports", "phase7_workbook_validation_log.txt");
const PREVIEW_DIR = path.join(EXP007, "reports", ".phase7_workbook_previews");

const FONT = "Arial";
const COLORS = {
  navy: "#1F4E78",
  blue: "#2F75B5",
  paleBlue: "#D9EAF7",
  paleGreen: "#E2F0D9",
  paleAmber: "#FFF2CC",
  paleRed: "#FCE4D6",
  greenText: "#375623",
  amberText: "#9C6500",
  redText: "#9C0006",
  white: "#FFFFFF",
  gray: "#666666",
};

function columnLetter(indexZeroBased) {
  let n = indexZeroBased + 1;
  let result = "";
  while (n > 0) {
    const remainder = (n - 1) % 26;
    result = String.fromCharCode(65 + remainder) + result;
    n = Math.floor((n - 1) / 26);
  }
  return result;
}

function matrixFromRecords(records) {
  if (!records || records.length === 0) return { headers: [], rows: [] };
  const headers = Object.keys(records[0]);
  return { headers, rows: records.map((record) => headers.map((header) => record[header] ?? null)) };
}

function selectRecords(records, headers) {
  return records.map((record) => Object.fromEntries(headers.map((header) => [header, record[header] ?? null])));
}

function styleTitle(sheet, title, note) {
  sheet.showGridLines = false;
  sheet.getRange("A2").values = [[title]];
  sheet.getRange("A2").format.font = { name: FONT, size: 15, bold: true, color: "#17365D" };
  sheet.getRange("A3:J3").format.fill = COLORS.navy;
  sheet.getRange("A3:J3").format.rowHeight = 3;
  sheet.getRange("A4").values = [[note]];
  sheet.getRange("A4").format.font = { name: FONT, size: 10, italic: true, color: COLORS.gray };
}

function widthForHeader(header) {
  const h = header.toLowerCase();
  if (h === "check" || h === "metric") return 42;
  if (h === "role") return 46;
  if (h === "item") return 30;
  if (h === "result") return 62;
  if (h === "definition" || h.includes("detail")) return 48;
  if (h.includes("display_name") || h.includes("scope_or_group")) return 34;
  if (h.includes("group_key")) return 36;
  if (h.includes("formula") || h.includes("material_name")) return 30;
  if (h.includes("doi") || h.includes("path")) return 30;
  if (h.includes("sha256")) return 29;
  if (h.includes("validation_") || h === "row_id") return 23;
  if (h.includes("interpretation") || h.includes("descriptive_result") || h.includes("support_note")) return 35;
  if (h.includes("metric_label")) return 25;
  if (h.includes("status") || h.includes("evaluation") || h.includes("weighting") || h.includes("dimension")) return 22;
  if (h.startsWith("n_") || h.endsWith("_rows") || h.endsWith("_samples") || h.endsWith("_count") || h.includes("rank")) return 14;
  if (h.includes("fraction") || h.includes("accuracy") || h.includes("probability")) return 18;
  if (h.includes("residual") || h.includes("rmse") || h.includes("mae") || h.includes("delta") || h.includes("log10")) return 19;
  return Math.min(25, Math.max(12, header.length + 2));
}

function formatColumn(sheet, header, index, firstDataRow, lastDataRow) {
  if (lastDataRow < firstDataRow) return;
  const letter = columnLetter(index);
  const range = sheet.getRange(`${letter}${firstDataRow}:${letter}${lastDataRow}`);
  const h = header.toLowerCase();
  if (h.includes("delta_percentage_points")) range.format.numberFormat = "0.00";
  else if (h.includes("fraction") || h.includes("accuracy") || h.includes("probability")) range.format.numberFormat = "0.0%";
  else if (h.includes("p_value") || h.endsWith("_p")) range.format.numberFormat = "0.000E+00";
  else if (h.startsWith("n_") || h.endsWith("_rows") || h.endsWith("_samples") || h.endsWith("_count") || h.includes("rank") || h.includes("replicates") || h.includes("seed")) range.format.numberFormat = "#,##0";
  else if (h.includes("sigma") && h.includes("per_m")) range.format.numberFormat = "0.000E+00";
  else if (h === "t_k") range.format.numberFormat = "0.0";
  else if (h.includes("residual") || h.includes("rmse") || h.includes("mae") || h.includes("delta") || h.includes("log10") || h.includes("ci95") || h.includes("point_delta")) range.format.numberFormat = "0.000";
}

function addTextFormat(range, text, fill, color) {
  range.conditionalFormats.add("containsText", { text, format: { fill, font: { bold: true, color } } });
}

function addStatusFormatting(sheet, headers, firstDataRow, lastDataRow) {
  if (lastDataRow < firstDataRow) return;
  headers.forEach((header, index) => {
    const h = header.toLowerCase();
    if (!(h.includes("status") || h.includes("evaluation") || h.includes("interpretation") || h.includes("result") || h.includes("support_note") || h.includes("error_change"))) return;
    const letter = columnLetter(index);
    const range = sheet.getRange(`${letter}${firstDataRow}:${letter}${lastDataRow}`);
    for (const text of ["PASS", "improved", "supported_improvement", "adequate"]) addTextFormat(range, text, COLORS.paleGreen, COLORS.greenText);
    for (const text of ["inconclusive", "mixed", "limited", "coverage_loss"]) addTextFormat(range, text, COLORS.paleAmber, COLORS.amberText);
    for (const text of ["FAIL", "worsened", "supported_worsening", "very_limited"]) addTextFormat(range, text, COLORS.paleRed, COLORS.redText);
  });
}

function writeTableSheet(workbook, { name, title, note, records, tableName, tabColor = null, freezeColumns = 2 }) {
  const sheet = workbook.worksheets.add(name);
  styleTitle(sheet, title, note);
  if (tabColor) sheet.tabColor = tabColor;
  const { headers, rows } = matrixFromRecords(records);
  if (headers.length === 0) {
    sheet.getRange("A6").values = [["No records"]];
    return { sheet, lastRow: 6, lastColumn: 0 };
  }
  const headerRow = 6;
  const firstDataRow = 7;
  const lastDataRow = firstDataRow + rows.length - 1;
  const lastColumn = headers.length - 1;
  const lastLetter = columnLetter(lastColumn);
  sheet.getRange(`A${headerRow}:${lastLetter}${headerRow}`).values = [headers];
  if (rows.length) sheet.getRange(`A${firstDataRow}:${lastLetter}${lastDataRow}`).values = rows;
  const header = sheet.getRange(`A${headerRow}:${lastLetter}${headerRow}`);
  header.format.fill = COLORS.navy;
  header.format.font = { name: FONT, size: 10, bold: true, color: COLORS.white };
  header.format.horizontalAlignment = "center";
  header.format.verticalAlignment = "center";
  header.format.wrapText = true;
  header.format.rowHeight = 36;
  if (rows.length) {
    const body = sheet.getRange(`A${firstDataRow}:${lastLetter}${lastDataRow}`);
    body.format.font = { name: FONT, size: 10, color: "#222222" };
    body.format.verticalAlignment = "center";
    body.format.rowHeight = 18;
    body.format.borders = { insideHorizontal: { style: "thin", color: "#E7E6E6" }, bottom: { style: "thin", color: "#B4C7E7" } };
    const table = sheet.tables.add(`A${headerRow}:${lastLetter}${lastDataRow}`, true, tableName);
    table.style = "TableStyleMedium2";
    table.showBandedRows = true;
    table.showFilterButton = true;
  }
  headers.forEach((headerName, index) => {
    const letter = columnLetter(index);
    sheet.getRange(`${letter}:${letter}`).format.columnWidth = widthForHeader(headerName);
    formatColumn(sheet, headerName, index, firstDataRow, lastDataRow);
  });
  addStatusFormatting(sheet, headers, firstDataRow, lastDataRow);
  for (const metric of ["delta_oxide_v1_minus_baseline", "row_delta_mae_log10", "sample_delta_mae_log10", "delta_abs_median_error_oxide_v1_minus_baseline"]) {
    const index = headers.indexOf(metric);
    if (index < 0 || rows.length < 2) continue;
    const letter = columnLetter(index);
    sheet.getRange(`${letter}${firstDataRow}:${letter}${lastDataRow}`).conditionalFormats.add("colorScale", {
      colors: ["#E2F0D9", "#FFFFFF", "#F4CCCC"],
      thresholds: ["min", { type: "percentile", value: 50 }, "max"],
    });
  }
  sheet.freezePanes.freezeRows(6);
  sheet.freezePanes.freezeColumns(Math.min(freezeColumns, headers.length));
  return { sheet, lastRow: Math.max(headerRow, lastDataRow), lastColumn };
}

function writeSmallTable(sheet, startRow, startColumn, title, records, headerFill = COLORS.navy) {
  const { headers, rows } = matrixFromRecords(records);
  const startLetter = columnLetter(startColumn);
  sheet.getRange(`${startLetter}${startRow}`).values = [[title]];
  sheet.getRange(`${startLetter}${startRow}`).format.font = { name: FONT, size: 12, bold: true, color: COLORS.navy };
  const headerRow = startRow + 2;
  const firstDataRow = headerRow + 1;
  const lastDataRow = firstDataRow + rows.length - 1;
  const lastColumn = startColumn + headers.length - 1;
  const lastLetter = columnLetter(lastColumn);
  sheet.getRange(`${startLetter}${headerRow}:${lastLetter}${headerRow}`).values = [headers];
  if (rows.length) sheet.getRange(`${startLetter}${firstDataRow}:${lastLetter}${lastDataRow}`).values = rows;
  const header = sheet.getRange(`${startLetter}${headerRow}:${lastLetter}${headerRow}`);
  header.format.fill = headerFill;
  header.format.font = { name: FONT, size: 10, bold: true, color: COLORS.white };
  header.format.horizontalAlignment = "center";
  header.format.verticalAlignment = "center";
  header.format.wrapText = true;
  header.format.rowHeight = 34;
  if (rows.length) {
    const body = sheet.getRange(`${startLetter}${firstDataRow}:${lastLetter}${lastDataRow}`);
    body.format.font = { name: FONT, size: 10, color: "#222222" };
    body.format.verticalAlignment = "center";
    body.format.borders = { insideHorizontal: { style: "thin", color: "#E7E6E6" } };
  }
  headers.forEach((headerName, relativeIndex) => {
    const index = startColumn + relativeIndex;
    const letter = columnLetter(index);
    sheet.getRange(`${letter}:${letter}`).format.columnWidth = widthForHeader(headerName);
    formatColumn(sheet, headerName, index, firstDataRow, lastDataRow);
  });
  addStatusFormatting(sheet, headers, firstDataRow, lastDataRow);
  return { lastRow: Math.max(headerRow, lastDataRow), lastColumn };
}

function writeSummarySheet(workbook, payload) {
  const sheet = workbook.worksheets.add("Summary");
  styleTitle(sheet, payload.title, payload.scope_note);
  sheet.tabColor = COLORS.navy;

  const decision = payload.tables.decision;
  writeSmallTable(sheet, 6, 0, "Decision summary", decision, COLORS.navy);
  sheet.getRange("A:A").format.columnWidthPx = 220;
  sheet.getRange("B:B").format.columnWidthPx = 380;
  sheet.getRange("C:C").format.columnWidthPx = 570;
  sheet.getRange("D:D").format.columnWidthPx = 28;

  sheet.getRange("E6").values = [["Main findings"]];
  sheet.getRange("E6").format.font = { name: FONT, size: 12, bold: true, color: COLORS.navy };
  payload.insights.forEach((text, index) => {
    sheet.getRange(`E${8 + index}`).values = [[text]];
    sheet.getRange(`E${8 + index}`).format.font = { name: FONT, size: 10, color: "#222222" };
  });
  sheet.getRange("E:E").format.columnWidthPx = 660;

  const primaryMetrics = selectRecords(
    payload.tables.paired_metrics.filter((record) => record.analysis_scope === "oxide_default_eligible" && record.metric_weighting === "row_equal"),
    ["metric_label", "baseline_value", "oxide_v1_value", "delta_oxide_v1_minus_baseline", "delta_percentage_points", "evaluation"],
  );
  const primaryBootstrap = selectRecords(
    payload.tables.bootstrap.filter((record) => record.analysis_scope === "oxide_default_eligible" && record.metric_weighting === "row_equal"),
    ["metric_label", "point_delta", "ci95_lower", "ci95_upper", "bootstrap_probability_improved", "interval_interpretation"],
  );
  const inventory = payload.tables.inventory;
  const metricsBlock = writeSmallTable(sheet, 21, 0, "Primary paired metrics", primaryMetrics, COLORS.blue);
  writeSmallTable(sheet, metricsBlock.lastRow + 3, 0, "Sample-cluster bootstrap intervals", primaryBootstrap, COLORS.blue);
  writeSmallTable(sheet, 21, 8, "Evaluation inventory", inventory, COLORS.navy);
  sheet.getRange("A:A").format.columnWidth = 30;
  sheet.getRange("B:B").format.columnWidth = 48;
  sheet.getRange("C:C").format.columnWidth = 70;
  sheet.getRange("D:D").format.columnWidth = 18;
  sheet.getRange("A9:C18").format.wrapText = true;
  sheet.getRange("A9:C18").format.rowHeight = 30;
  sheet.getRange("A9:C18").format.verticalAlignment = "top";
  sheet.getRange("I:I").format.columnWidthPx = 330;
  sheet.getRange("J:J").format.columnWidthPx = 130;
  sheet.getRange("K:K").format.columnWidthPx = 90;
  sheet.getRange("L:L").format.columnWidthPx = 520;
  sheet.freezePanes.freezeRows(4);
  return sheet;
}

function writeValidationSheet(workbook, payload) {
  const sheet = workbook.worksheets.add("Validation");
  styleTitle(sheet, "Phase 7 validation and input manifest", "Build checks are terminal audit evidence. The analysis does not depend on this sheet.");
  sheet.tabColor = "#7F8C8D";
  const checks = writeSmallTable(sheet, 6, 0, "Validation checks", payload.tables.validation, COLORS.navy);
  writeSmallTable(sheet, checks.lastRow + 4, 0, "Input manifest", payload.tables.manifest, COLORS.blue);
  sheet.freezePanes.freezeRows(6);
  return sheet;
}

async function main() {
  const payload = JSON.parse(await fs.readFile(PAYLOAD_PATH, "utf8"));
  const workbook = Workbook.create();
  const sheets = [];
  sheets.push(writeSummarySheet(workbook, payload));
  sheets.push(writeTableSheet(workbook, {
    name: "Primary metrics",
    title: "Paired metric evaluation",
    note: "Negative MAE/RMSE deltas favor oxide_v1. Positive accuracy deltas favor oxide_v1. Scope and weighting must be read together.",
    records: payload.tables.paired_metrics,
    tableName: "Phase7PairedMetrics",
    tabColor: COLORS.blue,
    freezeColumns: 4,
  }).sheet);
  sheets.push(writeTableSheet(workbook, {
    name: "Bootstrap CIs",
    title: "Sample-cluster bootstrap intervals",
    note: "Two thousand deterministic paired bootstrap replicates. Each held-out validation sample is one resampling cluster.",
    records: payload.tables.bootstrap,
    tableName: "Phase7Bootstrap",
    tabColor: COLORS.blue,
    freezeColumns: 4,
  }).sheet);
  sheets.push(writeTableSheet(workbook, {
    name: "Oxide groups",
    title: "Default-eligible oxide-group evaluation",
    note: "Error changes use paired rows. Coverage uses all available default-eligible test rows. Support notes flag small sample counts.",
    records: payload.tables.groups,
    tableName: "Phase7OxideGroups",
    freezeColumns: 3,
  }).sheet);
  sheets.push(writeTableSheet(workbook, {
    name: "Stratified",
    title: "Primary-cohort stratified evaluation",
    note: "The paired default-eligible oxide cohort is split by oxide group, carrier, 100 K bin, and observed conductivity decade.",
    records: payload.tables.stratified,
    tableName: "Phase7Stratified",
    freezeColumns: 3,
  }).sheet);
  sheets.push(writeTableSheet(workbook, {
    name: "Sample changes",
    title: "Default-eligible oxide sample changes",
    note: "Each row is one validation sample. Error is the absolute median log10 residual over the sample's paired temperature rows.",
    records: payload.tables.samples,
    tableName: "Phase7SampleChanges",
    freezeColumns: 4,
  }).sheet);
  sheets.push(writeTableSheet(workbook, {
    name: "Coverage losses",
    title: "Rows lost after oxide_v1 re-prediction",
    note: "These rows had a valid broad-oxide prediction but no valid oxide_v1 prediction because the finer reference bin was missing or unreliable.",
    records: payload.tables.coverage_losses,
    tableName: "Phase7CoverageLosses",
    freezeColumns: 4,
  }).sheet);
  sheets.push(writeTableSheet(workbook, {
    name: "Coverage summary",
    title: "Prediction coverage by scope and oxide group",
    note: "Coverage is calculated before paired-row filtering, so classification-induced availability loss remains visible.",
    records: payload.tables.coverage_summary,
    tableName: "Phase7CoverageSummary",
    freezeColumns: 3,
  }).sheet);
  sheets.push(writeValidationSheet(workbook, payload));

  workbook.recalculate();
  const summaryInspect = await workbook.inspect({
    kind: "table",
    range: "Summary!A2:O42",
    include: "values,formulas",
    tableMaxRows: 42,
    tableMaxCols: 15,
    maxChars: 12000,
  });
  const errors = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
    options: { useRegex: true, maxResults: 300 },
    summary: "final formula error scan",
  });

  await fs.mkdir(PREVIEW_DIR, { recursive: true });
  const previewFiles = [];
  const previewRanges = {
    Summary: "A1:O44",
    "Primary metrics": "A1:P50",
    "Bootstrap CIs": "A1:M48",
    "Oxide groups": "A1:AC18",
    Stratified: "A1:Q48",
    "Sample changes": "A1:Q40",
    "Coverage losses": "A1:S40",
    "Coverage summary": "A1:M30",
    Validation: "A1:D34",
  };
  for (const sheet of sheets) {
    const preview = await workbook.render({ sheetName: sheet.name, range: previewRanges[sheet.name], scale: 1, format: "png" });
    const previewPath = path.join(PREVIEW_DIR, `${sheet.name.replaceAll(" ", "_")}.png`);
    await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));
    previewFiles.push(previewPath);
  }

  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(OUTPUT_PATH);
  const errorText = errors.ndjson ?? "";
  const errorCount = (errorText.match(/#REF!|#DIV\/0!|#VALUE!|#NAME\?|#N\/A|#NUM!|#NULL!|#SPILL!|#CALC!/g) || []).length;
  const passed = errorCount === 0 && previewFiles.length === sheets.length;
  const lines = [
    "Phase 7 workbook validation",
    `status=${passed ? "PASS" : "FAIL"}`,
    `sheets=${sheets.length}`,
    `previews=${previewFiles.length}`,
    `formula_error_matches=${errorCount}`,
    `output_bytes=${(await fs.stat(OUTPUT_PATH)).size}`,
    "summary_inspect:",
    summaryInspect.ndjson ?? "",
    "formula_error_scan:",
    errorText,
  ];
  await fs.writeFile(VALIDATION_LOG, lines.join("\n") + "\n", "utf8");
  console.log(lines.slice(0, 6).join("\n"));
  if (!passed) process.exitCode = 1;
}

await main();
