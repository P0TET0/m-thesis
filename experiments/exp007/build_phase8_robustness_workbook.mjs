import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";


const ROOT = process.cwd();
const EXP007 = path.join(ROOT, "experiments", "exp007");
const PAYLOAD_PATH = path.join(EXP007, "reports", ".phase8_workbook_payload.json");
const OUTPUT_PATH = path.join(EXP007, "reports", "phase8_robustness_check.xlsx");
const VALIDATION_LOG = path.join(EXP007, "reports", "phase8_workbook_validation_log.txt");
const PREVIEW_DIR = path.join(EXP007, "reports", ".phase8_workbook_previews");

const FONT = "Arial";
const COLORS = {
  navy: "#17365D",
  blue: "#2F75B5",
  teal: "#2E8B57",
  paleBlue: "#D9EAF7",
  paleGreen: "#E2F0D9",
  paleAmber: "#FFF2CC",
  paleRed: "#FCE4D6",
  white: "#FFFFFF",
  gray: "#666666",
  greenText: "#375623",
  amberText: "#9C6500",
  redText: "#9C0006",
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

function styleTitle(sheet, title, note) {
  sheet.showGridLines = false;
  sheet.getRange("A2").values = [[title]];
  sheet.getRange("A2").format.font = { name: FONT, size: 16, bold: true, color: COLORS.navy };
  sheet.getRange("A3:J3").format.fill = COLORS.navy;
  sheet.getRange("A3:J3").format.rowHeight = 3;
  sheet.getRange("A4").values = [[note]];
  sheet.getRange("A4").format.font = { name: FONT, size: 10, italic: true, color: COLORS.gray };
}

function widthForHeader(header) {
  const h = header.toLowerCase();
  if (h === "item" || h === "metric" || h === "check") return 36;
  if (h === "result") return 58;
  if (h.includes("definition") || h.includes("detail") || h.includes("filter_rule")) return 48;
  if (h.includes("scenario_id") || h.includes("group_name") || h.includes("display_name")) return 33;
  if (h.includes("group_key")) return 38;
  if (h.includes("path") || h.includes("role")) return 48;
  if (h.includes("sha256")) return 28;
  if (h.includes("status") || h.includes("direction") || h.includes("weighting") || h.includes("profile") || h.includes("method")) return 23;
  if (h.startsWith("n_") || h.endsWith("_rows") || h.endsWith("_samples") || h.endsWith("_papers") || h.includes("count")) return 14;
  if (h.includes("fraction") || h.includes("accuracy")) return 18;
  if (h.includes("mae") || h.includes("rmse") || h.includes("delta") || h.includes("log10")) return 18;
  return Math.min(25, Math.max(12, header.length + 2));
}

function formatColumn(sheet, header, index, firstDataRow, lastDataRow) {
  if (lastDataRow < firstDataRow) return;
  const letter = columnLetter(index);
  const range = sheet.getRange(`${letter}${firstDataRow}:${letter}${lastDataRow}`);
  const h = header.toLowerCase();
  if (h.includes("fraction") || h.includes("accuracy")) range.format.numberFormat = "0.0%";
  else if (h.startsWith("n_") || h.endsWith("_rows") || h.endsWith("_samples") || h.endsWith("_papers") || h.includes("count") || h.includes("width_k") || h.startsWith("min_train")) range.format.numberFormat = "#,##0";
  else if (h.includes("mae") || h.includes("rmse") || h.includes("delta") || h.includes("log10") || h.includes("cutoff")) range.format.numberFormat = "0.000";
}

function addTextFormat(range, text, fill, color) {
  range.conditionalFormats.add("containsText", { text, format: { fill, font: { bold: true, color } } });
}

function addStatusFormatting(sheet, headers, firstDataRow, lastDataRow) {
  if (lastDataRow < firstDataRow) return;
  headers.forEach((header, index) => {
    const h = header.toLowerCase();
    if (!(h.includes("status") || h.includes("direction") || h.includes("result") || h.includes("verdict"))) return;
    const letter = columnLetter(index);
    const range = sheet.getRange(`${letter}${firstDataRow}:${letter}${lastDataRow}`);
    for (const text of ["PASS", "improved", "consistently_improved", "robust_with"]) addTextFormat(range, text, COLORS.paleGreen, COLORS.greenText);
    for (const text of ["mixed", "caveat", "not_external_validation"]) addTextFormat(range, text, COLORS.paleAmber, COLORS.amberText);
    for (const text of ["FAIL", "worsened", "consistently_worsened", "not_both_improved"]) addTextFormat(range, text, COLORS.paleRed, COLORS.redText);
  });
}

function writeTable(sheet, startRow, records, tableName) {
  const { headers, rows } = matrixFromRecords(records);
  if (!headers.length) {
    sheet.getRange(`A${startRow}`).values = [["No records"]];
    return { headers: [], lastRow: startRow, lastColumn: 0 };
  }
  const headerRow = startRow;
  const firstDataRow = startRow + 1;
  const lastDataRow = firstDataRow + rows.length - 1;
  const lastColumn = headers.length - 1;
  const lastLetter = columnLetter(lastColumn);
  sheet.getRange(`A${headerRow}:${lastLetter}${headerRow}`).values = [headers];
  if (rows.length) sheet.getRange(`A${firstDataRow}:${lastLetter}${lastDataRow}`).values = rows;
  const headerRange = sheet.getRange(`A${headerRow}:${lastLetter}${headerRow}`);
  headerRange.format.fill = COLORS.navy;
  headerRange.format.font = { name: FONT, size: 10, bold: true, color: COLORS.white };
  headerRange.format.horizontalAlignment = "center";
  headerRange.format.verticalAlignment = "center";
  headerRange.format.wrapText = true;
  headerRange.format.rowHeight = 36;
  if (rows.length) {
    const body = sheet.getRange(`A${firstDataRow}:${lastLetter}${lastDataRow}`);
    body.format.font = { name: FONT, size: 10, color: "#222222" };
    body.format.verticalAlignment = "center";
    body.format.rowHeight = 18;
    body.format.borders = { insideHorizontal: { style: "thin", color: "#E7E6E6" } };
    const table = sheet.tables.add(`A${headerRow}:${lastLetter}${lastDataRow}`, true, tableName);
    table.style = "TableStyleMedium2";
    table.showBandedRows = true;
    table.showFilterButton = true;
  }
  headers.forEach((header, index) => {
    const letter = columnLetter(index);
    sheet.getRange(`${letter}:${letter}`).format.columnWidth = widthForHeader(header);
    formatColumn(sheet, header, index, firstDataRow, lastDataRow);
  });
  addStatusFormatting(sheet, headers, firstDataRow, lastDataRow);
  for (const header of ["delta_mae_log10", "delta_rmse_log10", "median_delta_mae_log10"]) {
    const index = headers.indexOf(header);
    if (index < 0 || !rows.length) continue;
    const letter = columnLetter(index);
    sheet.getRange(`${letter}${firstDataRow}:${letter}${lastDataRow}`).conditionalFormats.add("colorScale", {
      colors: ["#E2F0D9", "#FFFFFF", "#F4CCCC"],
      thresholds: ["min", { type: "percentile", value: 50 }, "max"],
    });
  }
  return { headers, lastRow: Math.max(headerRow, lastDataRow), lastColumn };
}

function addTableSheet(workbook, { name, title, note, records, tableName, tabColor = COLORS.blue, freezeColumns = 2 }) {
  const sheet = workbook.worksheets.add(name);
  styleTitle(sheet, title, note);
  sheet.tabColor = tabColor;
  const table = writeTable(sheet, 6, records, tableName);
  sheet.freezePanes.freezeRows(6);
  sheet.freezePanes.freezeColumns(Math.min(freezeColumns, table.headers.length));
  return sheet;
}

function writeSummary(workbook, payload) {
  const sheet = workbook.worksheets.add("Summary");
  styleTitle(sheet, payload.title, payload.scope_note);
  sheet.tabColor = COLORS.navy;
  sheet.getRange("A6").values = [["Decision"]];
  sheet.getRange("A6").format.font = { name: FONT, size: 12, bold: true, color: COLORS.navy };
  const decision = writeTable(sheet, 8, payload.tables.decision, "Phase8Decision");
  const inventoryStart = decision.lastRow + 4;
  sheet.getRange(`A${inventoryStart - 2}`).values = [["Analysis inventory"]];
  sheet.getRange(`A${inventoryStart - 2}`).format.font = { name: FONT, size: 12, bold: true, color: COLORS.navy };
  writeTable(sheet, inventoryStart, payload.tables.inventory, "Phase8Inventory");
  sheet.getRange("A:A").format.columnWidth = 38;
  sheet.getRange("B:B").format.columnWidth = 65;
  sheet.getRange("C:C").format.columnWidth = 72;
  sheet.getRange("A9:C20").format.wrapText = true;
  sheet.getRange("A9:C20").format.rowHeight = 32;
  sheet.freezePanes.freezeRows(4);
  return sheet;
}

function writeValidation(workbook, payload) {
  const sheet = workbook.worksheets.add("Validation");
  styleTitle(sheet, "Phase 8 validation and sources", "Terminal checks confirm frozen-result reproduction, cohort integrity, and output completeness.");
  sheet.tabColor = "#7F8C8D";
  sheet.getRange("A6").values = [["Validation checks"]];
  sheet.getRange("A6").format.font = { name: FONT, size: 12, bold: true, color: COLORS.navy };
  const checks = writeTable(sheet, 8, payload.tables.validation, "Phase8Validation");
  sheet.getRange(`C9:C${checks.lastRow}`).format.columnWidthPx = 650;
  sheet.getRange(`C9:C${checks.lastRow}`).format.wrapText = true;
  sheet.getRange(`A9:C${checks.lastRow}`).format.rowHeight = 30;
  const manifestStart = checks.lastRow + 5;
  sheet.getRange(`A${manifestStart - 2}`).values = [["Input manifest"]];
  sheet.getRange(`A${manifestStart - 2}`).format.font = { name: FONT, size: 12, bold: true, color: COLORS.navy };
  const manifest = writeTable(sheet, manifestStart, payload.tables.manifest, "Phase8Manifest");
  sheet.getRange(`A${manifestStart + 1}:D${manifest.lastRow}`).format.wrapText = true;
  sheet.getRange(`A${manifestStart + 1}:D${manifest.lastRow}`).format.rowHeight = 32;
  sheet.getRange("A:A").format.columnWidthPx = 390;
  sheet.getRange("B:B").format.columnWidthPx = 520;
  sheet.getRange("C:C").format.columnWidthPx = 650;
  sheet.getRange("D:D").format.columnWidthPx = 560;
  sheet.freezePanes.freezeRows(8);
  return sheet;
}

async function main() {
  const payload = JSON.parse(await fs.readFile(PAYLOAD_PATH, "utf8"));
  const workbook = Workbook.create();
  const sheets = [];
  sheets.push(writeSummary(workbook, payload));
  sheets.push(addTableSheet(workbook, {
    name: "Model sensitivity",
    title: "Model-setting sensitivity",
    note: "Thirteen settings; each temperature-bin, support, or aggregation change is applied identically to broad oxide and oxide_v1. Negative error deltas favor oxide_v1.",
    records: payload.tables.model_metrics,
    tableName: "Phase8ModelSensitivity",
    freezeColumns: 7,
  }));
  sheets.push(addTableSheet(workbook, {
    name: "Evaluation sensitivity",
    title: "Weighting and extreme-row sensitivity",
    note: "Frozen 100 K setting. Symmetric trimming ranks rows by the larger absolute residual across the two variants, avoiding model-favoring removal.",
    records: payload.tables.evaluation,
    tableName: "Phase8EvaluationSensitivity",
    freezeColumns: 4,
  }));
  sheets.push(addTableSheet(workbook, {
    name: "Leave-one-group-out",
    title: "Leave-one-oxide-group-out influence check",
    note: "Each default-eligible oxide group is omitted in turn. Results are shown under row-equal and sample-equal weighting.",
    records: payload.tables.leave_one_out,
    tableName: "Phase8LeaveOneOut",
    freezeColumns: 4,
  }));
  sheets.push(addTableSheet(workbook, {
    name: "Group consistency",
    title: "Oxide-group direction across model settings",
    note: "This is descriptive: small groups remain sensitive even when the overall conclusion is robust.",
    records: payload.tables.group_consistency,
    tableName: "Phase8GroupConsistency",
    freezeColumns: 2,
  }));
  sheets.push(writeValidation(workbook, payload));

  workbook.recalculate();
  const summaryInspect = await workbook.inspect({
    kind: "table",
    range: "Summary!A2:F38",
    include: "values,formulas",
    tableMaxRows: 38,
    tableMaxCols: 6,
    maxChars: 12000,
  });
  const errors = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
    options: { useRegex: true, maxResults: 300 },
    summary: "final formula error scan",
  });

  await fs.mkdir(PREVIEW_DIR, { recursive: true });
  const previewRanges = {
    Summary: "A1:F38",
    "Model sensitivity": "A1:AG48",
    "Evaluation sensitivity": "A1:Z16",
    "Leave-one-group-out": "A1:X32",
    "Group consistency": "A1:K20",
    Validation: "A1:D38",
  };
  const previewFiles = [];
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
    "Phase 8 workbook validation",
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
