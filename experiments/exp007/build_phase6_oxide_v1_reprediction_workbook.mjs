import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";


const ROOT = process.cwd();
const EXP007 = path.join(ROOT, "experiments", "exp007");
const PAYLOAD_PATH = path.join(EXP007, "reports", ".phase6_workbook_payload.json");
const OUTPUT_PATH = path.join(EXP007, "reports", "phase6_oxide_v1_reprediction.xlsx");
const VALIDATION_LOG = path.join(EXP007, "reports", "phase6_workbook_validation_log.txt");
const PREVIEW_DIR = path.join(EXP007, "reports", ".phase6_workbook_previews");

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

function reorderRecords(records, preferredHeaders) {
  if (!records || records.length === 0) return records;
  const allHeaders = Object.keys(records[0]);
  const headers = [...preferredHeaders, ...allHeaders.filter((header) => !preferredHeaders.includes(header))];
  return selectRecords(records, headers);
}

function applyBaseStyle(sheet) {
  sheet.showGridLines = false;
  sheet.getRange("A1:BM120").format.font = { name: FONT, size: 10, color: "#222222" };
  sheet.getRange("A1:BM120").format.verticalAlignment = "center";
}

function styleTitle(sheet, title, note) {
  sheet.getRange("A2").values = [[title]];
  sheet.getRange("A2").format.font = { name: FONT, size: 15, bold: true, color: "#17365D" };
  sheet.getRange("A3:J3").format.fill = COLORS.navy;
  sheet.getRange("A3:J3").format.rowHeight = 3;
  sheet.getRange("A4").values = [[note]];
  sheet.getRange("A4").format.font = { name: FONT, size: 10, italic: true, color: COLORS.gray };
}

function widthForHeader(header) {
  const h = header.toLowerCase();
  if (h === "metric_name") return 24;
  if (h.includes("definition") || h.includes("detail") || h.includes("review_reasons")) return 42;
  if (h.includes("group_key") || h.includes("display_name")) return 31;
  if (h.includes("formula") || h.includes("material_name") || h.includes("material_label")) return 30;
  if (h.includes("doi") || h.includes("path")) return 28;
  if (h.includes("sha256")) return 28;
  if (h.includes("validation_") || h.includes("sample_key") || h === "row_id") return 21;
  if (h.includes("status") || h.includes("scope") || h.includes("basis") || h.includes("variant")) return 20;
  if (h.startsWith("n_") || h.endsWith("_rows") || h.endsWith("_count") || h.includes("rank")) return 13;
  if (h.includes("fraction") || h.includes("accuracy")) return 16;
  if (h.includes("residual") || h.includes("rmse") || h.includes("mae") || h.includes("delta")) return 18;
  return Math.min(23, Math.max(12, header.length + 2));
}

function formatColumn(sheet, header, index, firstDataRow, lastDataRow) {
  if (lastDataRow < firstDataRow) return;
  const letter = columnLetter(index);
  const range = sheet.getRange(`${letter}${firstDataRow}:${letter}${lastDataRow}`);
  const h = header.toLowerCase();
  if (h.includes("fraction") || h.includes("accuracy") || h.includes("coverage_delta")) range.format.numberFormat = "0.0%";
  else if (h.startsWith("n_") || h.endsWith("_rows") || h.endsWith("_count") || h.includes("rank")) range.format.numberFormat = "#,##0";
  else if (h.includes("sigma") && h.includes("per_m")) range.format.numberFormat = "0.000E+00";
  else if (h === "t_k" || h.includes("t_bin_center")) range.format.numberFormat = "0.0";
  else if (h.includes("residual") || h.includes("rmse") || h.includes("mae") || h.includes("delta_log10") || h.includes("log10_error")) range.format.numberFormat = "0.000";
}

function addTextFormat(range, text, fill, color) {
  range.conditionalFormats.add("containsText", { text, format: { fill, font: { bold: true, color } } });
}

function addStatusFormatting(sheet, headers, firstDataRow, lastDataRow) {
  if (lastDataRow < firstDataRow) return;
  headers.forEach((header, index) => {
    const h = header.toLowerCase();
    const letter = columnLetter(index);
    const range = sheet.getRange(`${letter}${firstDataRow}:${letter}${lastDataRow}`);
    if (h.includes("status") || h.includes("error_change")) {
      addTextFormat(range, "PASS", COLORS.paleGreen, COLORS.greenText);
      addTextFormat(range, "include", COLORS.paleGreen, COLORS.greenText);
      addTextFormat(range, "improved", COLORS.paleGreen, COLORS.greenText);
      addTextFormat(range, "review", COLORS.paleAmber, COLORS.amberText);
      addTextFormat(range, "unreliable", COLORS.paleAmber, COLORS.amberText);
      addTextFormat(range, "missing", COLORS.paleRed, COLORS.redText);
      addTextFormat(range, "exclude", COLORS.paleRed, COLORS.redText);
      addTextFormat(range, "worsened", COLORS.paleRed, COLORS.redText);
      addTextFormat(range, "FAIL", COLORS.paleRed, COLORS.redText);
    }
  });
}

function writeTableSheet(workbook, { name, title, note, records, tableName, tabColor = null, freezeColumns = 2 }) {
  const sheet = workbook.worksheets.add(name);
  applyBaseStyle(sheet);
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
  const headerRange = sheet.getRange(`A${headerRow}:${lastLetter}${headerRow}`);
  headerRange.format.fill = COLORS.navy;
  headerRange.format.font = { name: FONT, size: 10, bold: true, color: COLORS.white };
  headerRange.format.horizontalAlignment = "center";
  headerRange.format.wrapText = true;
  headerRange.format.rowHeight = 34;
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
  headers.forEach((header, index) => {
    const letter = columnLetter(index);
    sheet.getRange(`${letter}:${letter}`).format.columnWidth = widthForHeader(header);
    formatColumn(sheet, header, index, firstDataRow, lastDataRow);
  });
  addStatusFormatting(sheet, headers, firstDataRow, lastDataRow);
  for (const metric of ["mae_log10", "rmse_log10", "delta_abs_median_error_phase6_minus_baseline"]) {
    const index = headers.indexOf(metric);
    if (index < 0 || rows.length < 2) continue;
    const letter = columnLetter(index);
    sheet.getRange(`${letter}${firstDataRow}:${letter}${lastDataRow}`).conditionalFormats.add("colorScale", {
      colors: metric.startsWith("delta") ? ["#E2F0D9", "#FFFFFF", "#F4CCCC"] : ["#E2F0D9", "#FFF2CC", "#F4CCCC"],
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
  const endColumn = startColumn + headers.length - 1;
  const endLetter = columnLetter(endColumn);
  sheet.getRange(`${startLetter}${headerRow}:${endLetter}${headerRow}`).values = [headers];
  if (rows.length) sheet.getRange(`${startLetter}${firstDataRow}:${endLetter}${lastDataRow}`).values = rows;
  const header = sheet.getRange(`${startLetter}${headerRow}:${endLetter}${headerRow}`);
  header.format.fill = headerFill;
  header.format.font = { name: FONT, size: 10, bold: true, color: COLORS.white };
  header.format.horizontalAlignment = "center";
  header.format.wrapText = true;
  header.format.rowHeight = 34;
  headers.forEach((name, relativeIndex) => {
    const index = startColumn + relativeIndex;
    const letter = columnLetter(index);
    sheet.getRange(`${letter}:${letter}`).format.columnWidth = widthForHeader(name);
    formatColumn(sheet, name, index, firstDataRow, lastDataRow);
  });
  addStatusFormatting(sheet, headers, firstDataRow, lastDataRow);
  return { lastRow: Math.max(headerRow, lastDataRow), lastColumn: endColumn };
}

function writeSummarySheet(workbook, payload) {
  const sheet = workbook.worksheets.add("Summary");
  applyBaseStyle(sheet);
  styleTitle(sheet, payload.title, payload.scope_note);
  sheet.tabColor = COLORS.navy;

  const inventory = payload.tables.inventory;
  const inv = matrixFromRecords(inventory);
  sheet.getRange("A6:D6").values = [inv.headers];
  sheet.getRange(`A7:D${6 + inv.rows.length}`).values = inv.rows;
  sheet.getRange("A6:D6").format.fill = COLORS.navy;
  sheet.getRange("A6:D6").format.font = { name: FONT, size: 10, bold: true, color: COLORS.white };
  sheet.getRange("A:A").format.columnWidth = 56;
  sheet.getRange("B:B").format.columnWidth = 50;
  sheet.getRange("C:C").format.columnWidth = 12;
  sheet.getRange("D:D").format.columnWidth = 67;
  inventory.forEach((record, idx) => {
    const row = 7 + idx;
    if (["rows", "samples", "groups", "bins"].includes(String(record.unit))) sheet.getRange(`B${row}`).format.numberFormat = "#,##0";
    if (String(record.unit) === "sha256") {
      sheet.getRange(`B${row}`).format.font = { name: FONT, size: 8, color: "#222222" };
      sheet.getRange(`B${row}`).format.wrapText = true;
      sheet.getRange(`B${row}`).format.rowHeight = 46;
    }
  });

  sheet.getRange("F6:J6").format.fill = COLORS.blue;
  sheet.getRange("F6").values = [["Key findings"]];
  sheet.getRange("F6:J6").format.font = { name: FONT, size: 10, bold: true, color: COLORS.white };
  payload.insights.forEach((value, idx) => sheet.getRange(`F${7 + idx}`).values = [[`${idx + 1}. ${value}`]]);
  sheet.getRange(`F7:F${6 + payload.insights.length}`).format.wrapText = true;
  sheet.getRange(`F7:F${6 + payload.insights.length}`).format.rowHeight = 58;
  sheet.getRange("F:F").format.columnWidth = 73;
  sheet.getRange("G:J").format.columnWidth = 3;

  const overall = selectRecords(payload.tables.overall, ["prediction_variant", "metric_weighting", "n_source_rows", "mae_log10", "rmse_log10", "factor_2_accuracy", "factor_10_accuracy", "prediction_ok_rows", "coverage_fraction"]);
  const paired = selectRecords(payload.tables.paired, ["prediction_variant", "metric_weighting", "n_source_rows", "mae_log10", "rmse_log10", "factor_2_accuracy", "factor_10_accuracy"]);
  const deltas = selectRecords(payload.tables.deltas.filter((r) => r.metric_weighting === "row_equal"), ["comparison_basis", "metric_name", "baseline_value", "phase6_value", "delta_phase6_minus_baseline", "lower_or_higher_is_better"]);
  const overallEnd = writeSmallTable(sheet, 28, 0, "Available-as-is metrics", overall, COLORS.navy);
  const pairedEnd = writeSmallTable(sheet, 28, 10, "Paired common-row metrics", paired, COLORS.blue);
  const deltaEnd = writeSmallTable(sheet, Math.max(overallEnd.lastRow, pairedEnd.lastRow) + 3, 0, "Row-equal metric deltas", deltas, COLORS.navy);
  return { sheet, lastRow: deltaEnd.lastRow, lastColumn: Math.max(9, pairedEnd.lastColumn, deltaEnd.lastColumn) };
}

function writeValidationSheet(workbook, payload) {
  const sheet = workbook.worksheets.add("Validation");
  applyBaseStyle(sheet);
  styleTitle(sheet, "Validation and input manifest", "Build checks and SHA-256 hashes for the frozen Phase 0 and Phase 4 inputs.");
  sheet.tabColor = "#A5A5A5";
  const checks = matrixFromRecords(payload.tables.validation);
  const manifest = matrixFromRecords(payload.tables.manifest);
  const checkEnd = columnLetter(checks.headers.length - 1);
  const manifestStartColumn = 5;
  const manifestStart = columnLetter(manifestStartColumn);
  const manifestEnd = columnLetter(manifestStartColumn + manifest.headers.length - 1);
  sheet.getRange(`A6:${checkEnd}6`).values = [checks.headers];
  sheet.getRange(`A7:${checkEnd}${6 + checks.rows.length}`).values = checks.rows;
  sheet.getRange(`${manifestStart}4`).values = [["Input manifest"]];
  sheet.getRange(`${manifestStart}4`).format.font = { name: FONT, size: 11, bold: true, color: COLORS.navy };
  sheet.getRange(`${manifestStart}6:${manifestEnd}6`).values = [manifest.headers];
  sheet.getRange(`${manifestStart}7:${manifestEnd}${6 + manifest.rows.length}`).values = manifest.rows;
  for (const address of [`A6:${checkEnd}6`, `${manifestStart}6:${manifestEnd}6`]) {
    sheet.getRange(address).format.fill = COLORS.navy;
    sheet.getRange(address).format.font = { name: FONT, size: 10, bold: true, color: COLORS.white };
  }
  addTextFormat(sheet.getRange(`B7:B${6 + checks.rows.length}`), "PASS", COLORS.paleGreen, COLORS.greenText);
  addTextFormat(sheet.getRange(`B7:B${6 + checks.rows.length}`), "FAIL", COLORS.paleRed, COLORS.redText);
  sheet.getRange("A:A").format.columnWidth = 35;
  sheet.getRange("B:B").format.columnWidth = 12;
  sheet.getRange("C:C").format.columnWidth = 52;
  sheet.getRange(`${manifestStart}:${manifestStart}`).format.columnWidth = 48;
  sheet.getRange(`${columnLetter(manifestStartColumn + 1)}:${columnLetter(manifestStartColumn + 1)}`).format.columnWidth = 82;
  sheet.getRange(`${columnLetter(manifestStartColumn + 2)}:${columnLetter(manifestStartColumn + 2)}`).format.columnWidth = 18;
  sheet.getRange(`${columnLetter(manifestStartColumn + 3)}:${columnLetter(manifestStartColumn + 3)}`).format.columnWidth = 34;
  sheet.freezePanes.freezeRows(6);
  return { sheet, lastRow: Math.max(6 + checks.rows.length, 6 + manifest.rows.length), lastColumn: manifestStartColumn + manifest.headers.length - 1 };
}

async function main() {
  const payload = JSON.parse(await fs.readFile(PAYLOAD_PATH, "utf8"));
  const workbook = Workbook.create();
  const built = [];

  const scopes = reorderRecords(payload.tables.scopes, ["analysis_scope", "comparison_basis", "prediction_variant", "metric_weighting", "test_rows_in_scope", "prediction_ok_rows_in_scope", "coverage_fraction_in_scope", "mae_log10", "rmse_log10", "mean_log10_error", "factor_2_accuracy", "factor_10_accuracy"]);
  const groups = reorderRecords(payload.tables.groups, ["analysis_scope", "comparison_basis", "prediction_variant", "metric_weighting", "oxide_v1_display_name_ja", "oxide_v1_group_key", "test_rows_in_group_scope", "n_source_rows", "n_samples", "n_papers", "mae_log10", "rmse_log10", "mean_log10_error", "factor_2_accuracy", "factor_10_accuracy"]);
  const coverage = reorderRecords(payload.tables.coverage, ["oxide_v1_display_name_ja", "oxide_v1_group_key", "test_rows", "n_samples", "n_papers", "default_eligible_rows", "review_rows", "excluded_rows", "baseline_prediction_ok_rows", "phase6_prediction_ok_rows", "paired_prediction_rows", "baseline_coverage_fraction", "phase6_coverage_fraction", "coverage_delta_phase6_minus_baseline"]);
  const oxideSamples = payload.tables.samples.filter((r) => Boolean(r.is_phase0_broad_oxide));
  const oxideComparison = payload.tables.comparison.filter((r) => Boolean(r.is_phase0_broad_oxide));
  const heldoutAssignments = payload.tables.assignments.filter((r) => r.sample_holdout_split === "test");
  const samples = reorderRecords(oxideSamples, ["abs_error_change_rank", "sample_error_change", "oxide_v1_display_name_ja", "oxide_v1_group_key", "formula_raw", "material_name_raw", "validation_sample_group_id", "doi", "n_paired_rows", "baseline_median_residual_log10", "phase6_median_residual_log10", "delta_abs_median_error_phase6_minus_baseline", "delta_median_prediction_log10_phase6_minus_baseline"]);
  const comparison = selectRecords(oxideComparison, ["row_id", "validation_sample_group_id", "validation_paper_group_id", "doi", "formula_raw", "material_name_raw", "oxide_v1_display_name_ja", "oxide_v1_group_key", "oxide_v1_inclusion_status", "carrier_type", "T_K", "T_bin_label", "sigma_S_per_m", "baseline_prediction_status", "phase6_prediction_status", "baseline_sigma_pred_S_per_m", "phase6_sigma_pred_S_per_m", "baseline_residual_log10", "phase6_residual_log10", "delta_log10_prediction_phase6_minus_baseline", "delta_abs_error_phase6_minus_baseline", "phase6_error_change"]);
  const assignments = selectRecords(heldoutAssignments, ["row_id", "validation_sample_group_id", "validation_paper_group_id", "sample_holdout_split", "formula_raw", "material_name_raw", "material_label_normalized", "parsed_elements", "oxide_v1_display_name_ja", "oxide_v1_group_key", "oxide_v1_inclusion_status", "oxide_v1_default_model_eligible", "oxide_v1_classification_confidence", "oxide_v1_review_priority", "oxide_v1_review_reasons"]);
  const reference = selectRecords(payload.tables.reference, ["material_group_key", "carrier_type", "T_bin_label", "T_bin_center_K", "sigma0_ref_S_per_m", "log10_sigma0_ref_S_per_m", "train_row_count", "train_sample_count", "train_paper_count", "is_reference_bin_candidate", "reliability_level", "reference_source_subset", "curve_method"]);

  built.push(writeSummarySheet(workbook, payload));
  built.push(writeTableSheet(workbook, { name: "Oxide scopes", title: "Oxide scope comparison", note: "Available-as-is and paired metrics for all assigned, default-eligible, review-only, and excluded oxide cohorts.", records: scopes, tableName: "Phase6Scopes", tabColor: COLORS.blue }));
  built.push(writeTableSheet(workbook, { name: "Oxide groups", title: "oxide_v1 group metrics", note: "Metrics by oxide_v1 group for all-assigned and default-eligible scopes. Paired rows isolate the prediction change.", records: groups, tableName: "Phase6Groups", tabColor: COLORS.blue }));
  built.push(writeTableSheet(workbook, { name: "Group coverage", title: "Prediction coverage by oxide_v1 group", note: "Coverage is computed against the unchanged held-out test rows in each oxide_v1 group.", records: coverage, tableName: "Phase6Coverage", tabColor: COLORS.blue }));
  built.push(writeTableSheet(workbook, { name: "Sample changes", title: "Oxide-sample prediction changes", note: "One row per oxide sample predicted by both variants. Negative absolute-error delta indicates improvement. The full all-material table is retained as CSV.", records: samples, tableName: "Phase6Samples" }));
  built.push(writeTableSheet(workbook, { name: "Prediction rows", title: "Baseline and Phase 6 oxide-row predictions", note: "The 3,437 frozen held-out oxide rows. The full 19,008-row comparison remains available as canonical CSV.", records: comparison, tableName: "Phase6Rows" }));
  built.push(writeTableSheet(workbook, { name: "Oxide assignments", title: "oxide_v1 assignments for held-out oxide rows", note: "All 3,437 held-out rows previously assigned to broad::oxide. The full 17,715-row train-and-test assignment table remains available as canonical CSV.", records: assignments, tableName: "Phase6Assignments" }));
  built.push(writeTableSheet(workbook, { name: "Reference bins", title: "Phase 6 train-only reference bins", note: "100 K sample-median sigma0 reference bins built from conservative-valid training rows only.", records: reference, tableName: "Phase6Reference" }));
  built.push(writeValidationSheet(workbook, payload));

  workbook.recalculate();
  await fs.mkdir(path.dirname(OUTPUT_PATH), { recursive: true });
  await fs.mkdir(PREVIEW_DIR, { recursive: true });
  const inspectSummary = await workbook.inspect({ kind: "table", range: "Summary!A1:R55", include: "values,formulas", tableMaxRows: 55, tableMaxCols: 18, maxChars: 20000 });
  const inspectCoverage = await workbook.inspect({ kind: "table", range: "Group coverage!A1:N22", include: "values,formulas", tableMaxRows: 22, tableMaxCols: 14, maxChars: 16000 });
  const errors = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!", options: { useRegex: true, maxResults: 300 }, summary: "final formula error scan", maxChars: 12000 });

  for (const entry of built) {
    const range = `A1:${columnLetter(Math.min(entry.lastColumn, 13))}${Math.min(entry.lastRow, 44)}`;
    const preview = await workbook.render({ sheetName: entry.sheet.name, range, scale: 1, format: "png" });
    const safeName = entry.sheet.name.replaceAll(" ", "_").toLowerCase();
    await fs.writeFile(path.join(PREVIEW_DIR, `${safeName}.png`), new Uint8Array(await preview.arrayBuffer()));
  }

  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(OUTPUT_PATH);
  const stat = await fs.stat(OUTPUT_PATH);
  const errorText = errors.ndjson ?? "";
  const hasErrors = /#REF!|#DIV\/0!|#VALUE!|#NAME\?|#N\/A|#NUM!|#NULL!|#SPILL!|#CALC!/.test(errorText);
  const validationText = [
    "Phase 6 workbook validation",
    `status=${hasErrors ? "FAIL" : "PASS"}`,
    `output=${path.relative(ROOT, OUTPUT_PATH).replaceAll("\\", "/")}`,
    `size_bytes=${stat.size}`,
    `sheets=${built.map((entry) => entry.sheet.name).join(" | ")}`,
    `formula_error_scan=${hasErrors ? "FAIL" : "PASS"}`,
    "summary_inspect:", inspectSummary.ndjson ?? "",
    "group_coverage_inspect:", inspectCoverage.ndjson ?? "",
    "formula_error_scan_output:", errorText,
  ].join("\n");
  await fs.writeFile(VALIDATION_LOG, validationText + "\n", "utf8");
  await fs.rm(PAYLOAD_PATH, { force: true });
  console.log(`Workbook written: ${OUTPUT_PATH}`);
  console.log(`Workbook validation: ${hasErrors ? "FAIL" : "PASS"}`);
}

await main();
