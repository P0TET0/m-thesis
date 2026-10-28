import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";


const ROOT = process.cwd();
const EXP007 = path.join(ROOT, "experiments", "exp007");
const PAYLOAD_PATH = path.join(EXP007, "reports", ".phase5_workbook_payload.json");
const OUTPUT_PATH = path.join(EXP007, "reports", "phase5_oxide_v1_reaggregation.xlsx");
const INSPECT_PATH = `${OUTPUT_PATH}.inspect.ndjson`;
const VALIDATION_LOG = path.join(EXP007, "reports", "phase5_workbook_validation_log.txt");
const PREVIEW_DIR = path.join(EXP007, "reports", ".phase5_workbook_previews");

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
  blueText: "#1F4E78",
  gray: "#666666",
  white: "#FFFFFF",
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
  return {
    headers,
    rows: records.map((record) => headers.map((header) => record[header] ?? null)),
  };
}

function reorderRecords(records, preferredHeaders) {
  if (!records || records.length === 0) return records;
  const allHeaders = Object.keys(records[0]);
  const ordered = [...preferredHeaders, ...allHeaders.filter((header) => !preferredHeaders.includes(header))];
  return records.map((record) => Object.fromEntries(ordered.map((header) => [header, record[header] ?? null])));
}

function selectRecords(records, headers) {
  return records.map((record) => Object.fromEntries(headers.map((header) => [header, record[header] ?? null])));
}

function applyBaseStyle(sheet) {
  sheet.showGridLines = false;
  sheet.getRange("A1:BM5000").format.font = { name: FONT, size: 10, color: "#222222" };
  sheet.getRange("A1:BM5000").format.verticalAlignment = "center";
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
  if (h.includes("definition") || h.includes("reason") || h.includes("detail") || h.includes("scope_definition")) return 40;
  if (h.includes("group_key") || h.includes("display_name")) return 30;
  if (h.includes("formula") || h.includes("material_name") || h.includes("material_label")) return 30;
  if (h.includes("path") || h.includes("doi")) return 28;
  if (h.includes("sha256")) return 25;
  if (h.includes("sample_key") || h.includes("validation_") || h === "row_id") return 20;
  if (h.includes("direction") || h.includes("action") || h.includes("scope")) return 20;
  if (h.includes("priority") || h.includes("status") || h.includes("confidence")) return 15;
  if (h.startsWith("n_") || h.endsWith("_count") || h.includes("rank")) return 13;
  if (h.includes("fraction") || h.includes("accuracy") || h.includes("share")) return 15;
  if (h.includes("residual") || h.includes("rmse") || h.includes("mae") || h.includes("bias")) return 17;
  return Math.min(22, Math.max(12, header.length + 2));
}

function formatColumn(sheet, header, index, firstDataRow, lastDataRow) {
  if (lastDataRow < firstDataRow) return;
  const letter = columnLetter(index);
  const range = sheet.getRange(`${letter}${firstDataRow}:${letter}${lastDataRow}`);
  const h = header.toLowerCase();
  if (h.includes("fraction") || h.includes("accuracy") || h.includes("share")) {
    range.format.numberFormat = "0.0%";
  } else if (h.startsWith("n_") || h.endsWith("_count") || h.includes("rank")) {
    range.format.numberFormat = "#,##0";
  } else if (h.includes("sigma_") && h.includes("per_m")) {
    range.format.numberFormat = "0.000E+00";
  } else if (h === "t_k" || h.includes("t_min") || h.includes("t_max") || h.includes("t_bin_center")) {
    range.format.numberFormat = "0.0";
  } else if (h.includes("residual") || h.includes("rmse") || h.includes("mae") || h.includes("bias")) {
    range.format.numberFormat = "0.000";
  } else if (h.includes("absolute_error_factor")) {
    range.format.numberFormat = "0.00E+00";
  } else if (h.includes("factor")) {
    range.format.numberFormat = "0.00";
  }
}

function addTextFormat(range, text, fill, color) {
  range.conditionalFormats.add("containsText", {
    text,
    format: { fill, font: { bold: true, color } },
  });
}

function addStatusFormatting(sheet, headers, firstDataRow, lastDataRow) {
  if (lastDataRow < firstDataRow) return;
  for (const statusHeader of ["oxide_v1_inclusion_status", "status"]) {
    const index = headers.indexOf(statusHeader);
    if (index < 0) continue;
    const letter = columnLetter(index);
    const range = sheet.getRange(`${letter}${firstDataRow}:${letter}${lastDataRow}`);
    addTextFormat(range, "include", COLORS.paleGreen, COLORS.greenText);
    addTextFormat(range, "PASS", COLORS.paleGreen, COLORS.greenText);
    addTextFormat(range, "review", COLORS.paleAmber, COLORS.amberText);
    addTextFormat(range, "exclude", COLORS.paleRed, COLORS.redText);
    addTextFormat(range, "FAIL", COLORS.paleRed, COLORS.redText);
  }
  for (const directionHeader of ["row_bias_direction", "sample_equal_bias_direction", "sample_prediction_direction", "prediction_direction"]) {
    const index = headers.indexOf(directionHeader);
    if (index < 0) continue;
    const letter = columnLetter(index);
    const range = sheet.getRange(`${letter}${firstDataRow}:${letter}${lastDataRow}`);
    addTextFormat(range, "overprediction", COLORS.paleRed, COLORS.redText);
    addTextFormat(range, "underprediction", COLORS.paleBlue, COLORS.blueText);
  }
}

function writeTableSheet(workbook, config) {
  const { name, title, note, records, tableName, tabColor = null } = config;
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
  const header = sheet.getRange(`A${headerRow}:${lastLetter}${headerRow}`);
  header.format.fill = COLORS.navy;
  header.format.font = { name: FONT, size: 10, bold: true, color: COLORS.white };
  header.format.horizontalAlignment = "center";
  header.format.wrapText = true;
  header.format.rowHeight = 34;

  if (rows.length) {
    const body = sheet.getRange(`A${firstDataRow}:${lastLetter}${lastDataRow}`);
    body.format.rowHeight = 18;
    body.format.borders = {
      insideHorizontal: { style: "thin", color: "#E7E6E6" },
      bottom: { style: "thin", color: "#B4C7E7" },
    };
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
  for (const metricHeader of ["sample_equal_mae_log10", "row_mae_log10", "mae_log10"]) {
    const index = headers.indexOf(metricHeader);
    if (index < 0 || rows.length < 2) continue;
    const letter = columnLetter(index);
    sheet.getRange(`${letter}${firstDataRow}:${letter}${lastDataRow}`).conditionalFormats.add("colorScale", {
      colors: ["#E2F0D9", "#FFF2CC", "#F4CCCC"],
      thresholds: ["min", { type: "percentile", value: 50 }, "max"],
    });
  }
  sheet.freezePanes.freezeRows(6);
  sheet.freezePanes.freezeColumns(Math.min(2, headers.length));
  return { sheet, lastRow: Math.max(headerRow, lastDataRow), lastColumn };
}

function writeSummarySheet(workbook, payload) {
  const sheet = workbook.worksheets.add("Summary");
  applyBaseStyle(sheet);
  styleTitle(sheet, payload.title, payload.scope_note);
  sheet.tabColor = COLORS.navy;

  const summary = payload.tables.summary;
  sheet.getRange("A6:D6").values = [["metric", "value", "unit", "definition"]];
  sheet.getRange(`A7:D${6 + summary.length}`).values = summary.map((r) => [r.metric, r.value, r.unit, r.definition]);
  sheet.getRange("A6:D6").format.fill = COLORS.navy;
  sheet.getRange("A6:D6").format.font = { name: FONT, size: 10, bold: true, color: COLORS.white };
  sheet.getRange("A:A").format.columnWidth = 42;
  sheet.getRange("B:B").format.columnWidth = 42;
  sheet.getRange("C:C").format.columnWidth = 12;
  sheet.getRange("D:D").format.columnWidth = 55;
  summary.forEach((record, idx) => {
    const row = 7 + idx;
    const unit = String(record.unit ?? "");
    if (unit === "fraction") sheet.getRange(`B${row}`).format.numberFormat = "0.0%";
    else if (unit === "log10") sheet.getRange(`B${row}`).format.numberFormat = "0.000";
    else if (["rows", "samples", "papers"].includes(unit)) sheet.getRange(`B${row}`).format.numberFormat = "#,##0";
    if (String(record.metric).includes("sha256")) {
      sheet.getRange(`B${row}`).format.font = { name: FONT, size: 8, color: "#222222" };
      sheet.getRange(`B${row}`).format.wrapText = true;
      sheet.getRange(`B${row}`).format.rowHeight = 30;
    }
  });

  sheet.getRange("F6:J6").format.fill = COLORS.blue;
  sheet.getRange("F6").values = [["Key findings"]];
  sheet.getRange("F6:J6").format.font = { name: FONT, size: 10, bold: true, color: COLORS.white };
  payload.insights.forEach((text, idx) => sheet.getRange(`F${7 + idx}`).values = [[`${idx + 1}. ${text}`]]);
  sheet.getRange(`F7:F${6 + payload.insights.length}`).format.wrapText = true;
  sheet.getRange(`F7:F${6 + payload.insights.length}`).format.rowHeight = 64;
  sheet.getRange("F:F").format.columnWidth = 76;
  sheet.getRange("G:J").format.columnWidth = 3;

  const compactHeaders = [
    "oxide_v1_display_name_ja", "n_samples", "n_rows",
    "sample_equal_mae_log10", "sample_equal_mean_residual_log10",
    "sample_equal_bias_direction", "row_within_factor_2_fraction",
    "row_within_factor_10_fraction", "descriptive_support_minimum_30rows_3samples_3papers",
  ];
  const compact = selectRecords(payload.tables.primary_groups, compactHeaders);
  const startRow = 29;
  const matrix = matrixFromRecords(compact);
  const endRow = startRow + matrix.rows.length;
  const endLetter = columnLetter(matrix.headers.length - 1);
  sheet.getRange(`A${startRow - 2}`).values = [["Default-eligible group comparison"]];
  sheet.getRange(`A${startRow - 2}`).format.font = { name: FONT, size: 12, bold: true, color: COLORS.navy };
  sheet.getRange(`A${startRow}:${endLetter}${startRow}`).values = [matrix.headers];
  sheet.getRange(`A${startRow + 1}:${endLetter}${endRow}`).values = matrix.rows;
  sheet.getRange(`A${startRow}:${endLetter}${startRow}`).format.fill = COLORS.navy;
  sheet.getRange(`A${startRow}:${endLetter}${startRow}`).format.font = { name: FONT, size: 10, bold: true, color: COLORS.white };
  sheet.getRange(`A${startRow}:${endLetter}${startRow}`).format.wrapText = true;
  sheet.getRange(`A${startRow}:${endLetter}${startRow}`).format.rowHeight = 34;
  matrix.headers.forEach((header, index) => {
    const letter = columnLetter(index);
    sheet.getRange(`${letter}:${letter}`).format.columnWidth = widthForHeader(header);
    formatColumn(sheet, header, index, startRow + 1, endRow);
  });
  addStatusFormatting(sheet, matrix.headers, startRow + 1, endRow);
  const maeIndex = matrix.headers.indexOf("sample_equal_mae_log10");
  if (maeIndex >= 0) {
    const letter = columnLetter(maeIndex);
    sheet.getRange(`${letter}${startRow + 1}:${letter}${endRow}`).conditionalFormats.add("colorScale", {
      colors: ["#E2F0D9", "#FFF2CC", "#F4CCCC"],
      thresholds: ["min", { type: "percentile", value: 50 }, "max"],
    });
  }
  return { sheet, lastRow: endRow, lastColumn: Math.max(9, matrix.headers.length - 1) };
}

function writeValidationSheet(workbook, payload) {
  const sheet = workbook.worksheets.add("Validation");
  applyBaseStyle(sheet);
  styleTitle(sheet, "Validation and input manifest", "Independent build checks and SHA-256 hashes for the frozen Phase 1 and Phase 4 inputs.");
  sheet.tabColor = "#A5A5A5";
  const checks = matrixFromRecords(payload.tables.validation);
  const manifest = matrixFromRecords(payload.tables.manifest);
  const checkEnd = columnLetter(checks.headers.length - 1);
  const manifestEnd = columnLetter(5 + manifest.headers.length - 1);
  sheet.getRange(`A6:${checkEnd}6`).values = [checks.headers];
  sheet.getRange(`A7:${checkEnd}${6 + checks.rows.length}`).values = checks.rows;
  sheet.getRange("F4").values = [["Input manifest"]];
  sheet.getRange("F4").format.font = { name: FONT, size: 11, bold: true, color: COLORS.navy };
  sheet.getRange(`F6:${manifestEnd}6`).values = [manifest.headers];
  sheet.getRange(`F7:${manifestEnd}${6 + manifest.rows.length}`).values = manifest.rows;
  for (const address of [`A6:${checkEnd}6`, `F6:${manifestEnd}6`]) {
    sheet.getRange(address).format.fill = COLORS.navy;
    sheet.getRange(address).format.font = { name: FONT, size: 10, bold: true, color: COLORS.white };
  }
  addTextFormat(sheet.getRange(`B7:B${6 + checks.rows.length}`), "PASS", COLORS.paleGreen, COLORS.greenText);
  addTextFormat(sheet.getRange(`B7:B${6 + checks.rows.length}`), "FAIL", COLORS.paleRed, COLORS.redText);
  sheet.getRange("A:A").format.columnWidth = 34;
  sheet.getRange("B:B").format.columnWidth = 12;
  sheet.getRange("C:C").format.columnWidth = 48;
  sheet.getRange("F:F").format.columnWidth = 30;
  sheet.getRange("G:G").format.columnWidth = 54;
  sheet.getRange("H:H").format.columnWidth = 16;
  sheet.getRange("I:I").format.columnWidth = 28;
  sheet.freezePanes.freezeRows(6);
  return { sheet, lastRow: Math.max(6 + checks.rows.length, 6 + manifest.rows.length), lastColumn: 8 };
}

async function main() {
  const payload = JSON.parse(await fs.readFile(PAYLOAD_PATH, "utf8"));
  const workbook = Workbook.create();
  const built = [];

  const primary = reorderRecords(payload.tables.primary_groups, [
    "oxide_v1_display_name_ja", "oxide_v1_group_key", "n_samples", "n_rows", "n_papers",
    "descriptive_support_minimum_30rows_3samples_3papers", "sample_equal_mae_log10",
    "sample_equal_mean_residual_log10", "sample_equal_bias_direction",
    "sample_equal_within_factor_2_fraction", "sample_equal_within_factor_10_fraction",
    "row_mae_log10", "row_mean_residual_log10", "row_bias_direction",
    "row_within_factor_2_fraction", "row_within_factor_10_fraction",
    "sample_equal_mae_high_error_rank", "sample_equal_abs_bias_rank",
  ]);
  const scopes = reorderRecords(payload.tables.overall_scopes, [
    "phase5_analysis_scope", "scope_definition", "n_rows", "n_samples", "n_papers",
    "row_mae_log10", "sample_equal_mae_log10", "row_mean_residual_log10",
    "sample_equal_mean_residual_log10", "row_within_factor_2_fraction",
    "row_within_factor_10_fraction", "sample_equal_within_factor_2_fraction",
    "sample_equal_within_factor_10_fraction",
  ]);
  const allScopes = reorderRecords(payload.tables.all_group_scopes, [
    "phase5_analysis_scope", "oxide_v1_display_name_ja", "oxide_v1_group_key",
    "n_samples", "n_rows", "n_papers", "sample_equal_mae_log10",
    "sample_equal_mean_residual_log10", "sample_equal_bias_direction",
    "row_mae_log10", "row_mean_residual_log10", "row_bias_direction",
    "row_within_factor_2_fraction", "row_within_factor_10_fraction",
    "descriptive_support_minimum_30rows_3samples_3papers",
  ]);
  const carrier = reorderRecords(payload.tables.carrier, [
    "oxide_v1_display_name_ja", "oxide_v1_group_key", "carrier_type",
    "n_samples", "n_rows", "n_papers", "sample_equal_mae_log10",
    "sample_equal_mean_residual_log10", "sample_equal_bias_direction",
    "row_mae_log10", "row_within_factor_2_fraction", "row_within_factor_10_fraction",
    "descriptive_support_minimum_30rows_3samples_3papers",
  ]);
  const temperature = reorderRecords(payload.tables.temperature, [
    "oxide_v1_display_name_ja", "oxide_v1_group_key", "T_bin_label", "T_bin_center_K",
    "n_samples", "n_rows", "n_papers", "sample_equal_mae_log10",
    "sample_equal_mean_residual_log10", "sample_equal_bias_direction",
    "row_mae_log10", "row_within_factor_2_fraction", "row_within_factor_10_fraction",
    "descriptive_support_minimum_30rows_3samples_3papers",
  ]);
  const outliers = reorderRecords(payload.tables.outliers, [
    "sample_abs_median_residual_rank", "oxide_v1_inclusion_status", "oxide_v1_display_name_ja",
    "formula_raw", "material_name_raw", "validation_sample_group_id", "doi", "n_rows",
    "median_residual_log10", "sample_absolute_error_factor", "sample_prediction_direction",
    "mae_log10", "max_abs_residual_log10", "worst_row_T_K", "worst_row_residual_log10",
    "oxide_v1_review_priority", "oxide_v1_review_reasons",
  ]);
  const samples = reorderRecords(payload.tables.samples, [
    "sample_abs_median_residual_rank", "oxide_v1_inclusion_status", "oxide_v1_display_name_ja",
    "oxide_v1_group_key", "formula_raw", "material_name_raw", "validation_sample_group_id",
    "doi", "n_rows", "median_residual_log10", "sample_absolute_error_factor",
    "sample_prediction_direction", "mae_log10", "max_abs_residual_log10",
    "sample_level_within_factor_2", "sample_level_within_factor_10",
  ]);
  const rows = reorderRecords(payload.tables.rows, [
    "oxide_v1_inclusion_status", "oxide_v1_display_name_ja", "oxide_v1_group_key",
    "formula_raw", "material_name_raw", "row_id", "validation_sample_group_id", "doi",
    "carrier_type", "T_K", "sigma_observed_S_per_m", "sigma_predicted_S_per_m",
    "residual_log10_pred_over_observed", "abs_residual_log10", "prediction_direction",
    "residual_band", "within_factor_2", "within_factor_10",
  ]);

  built.push(writeSummarySheet(workbook, payload));
  built.push(writeTableSheet(workbook, { name: "Primary groups", title: "Default-eligible oxide_v1 group metrics", note: "Primary comparison using the 553 default-eligible samples. Row-equal and sample-equal metrics describe existing predictions.", records: primary, tableName: "Phase5PrimaryGroups", tabColor: COLORS.blue }));
  built.push(writeTableSheet(workbook, { name: "Scope comparison", title: "Overall metrics by inclusion scope", note: "All assigned, default-eligible, review-only, and excluded-only cohorts are shown separately.", records: scopes, tableName: "Phase5Scopes", tabColor: COLORS.blue }));
  built.push(writeTableSheet(workbook, { name: "All group scopes", title: "Group metrics under every inclusion scope", note: "Long-format sensitivity table. Empty default-eligible cells identify groups withheld from the primary analysis.", records: allScopes, tableName: "Phase5AllScopes" }));
  built.push(writeTableSheet(workbook, { name: "Carrier metrics", title: "Default-eligible metrics by oxide_v1 group and carrier type", note: "Stratified existing-prediction metrics; small cells remain descriptive and are flagged by the support column.", records: carrier, tableName: "Phase5Carrier" }));
  built.push(writeTableSheet(workbook, { name: "Temperature metrics", title: "Default-eligible metrics by oxide_v1 group and 100 K bin", note: "Temperature bins are inherited from the frozen broad_family_100K baseline.", records: temperature, tableName: "Phase5Temperature" }));
  built.push(writeTableSheet(workbook, { name: "Outlier samples", title: "Samples with a factor-10 row or sample-median error", note: "A sample enters this table if its median absolute error exceeds factor 10 or any retained row exceeds factor 10.", records: outliers, tableName: "Phase5Outliers", tabColor: "#C55A11" }));
  built.push(writeTableSheet(workbook, { name: "Sample metrics", title: "Existing-prediction metrics for all oxide_v1 samples", note: "One row per validation sample. The median signed residual is the input to sample-equal group metrics.", records: samples, tableName: "Phase5Samples" }));
  built.push(writeTableSheet(workbook, { name: "Row predictions", title: "Phase 1 prediction rows with oxide_v1 assignments", note: "All 3,437 broad-oxide held-out rows. No prediction values were changed in Phase 5.", records: rows, tableName: "Phase5Rows" }));
  built.push(writeValidationSheet(workbook, payload));

  workbook.recalculate();
  await fs.mkdir(path.dirname(OUTPUT_PATH), { recursive: true });
  await fs.mkdir(PREVIEW_DIR, { recursive: true });
  const inspectSummary = await workbook.inspect({ kind: "table", range: "Summary!A1:J44", include: "values,formulas", tableMaxRows: 44, tableMaxCols: 10, maxChars: 16000 });
  const inspectPrimary = await workbook.inspect({ kind: "table", range: "Primary groups!A1:P21", include: "values,formulas", tableMaxRows: 21, tableMaxCols: 16, maxChars: 16000 });
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
    "Phase 5 workbook validation",
    `status=${hasErrors ? "FAIL" : "PASS"}`,
    `output=${path.relative(ROOT, OUTPUT_PATH).replaceAll("\\", "/")}`,
    `size_bytes=${stat.size}`,
    `sheets=${built.map((entry) => entry.sheet.name).join(" | ")}`,
    `formula_error_scan=${hasErrors ? "FAIL" : "PASS"}`,
    "summary_inspect:", inspectSummary.ndjson ?? "",
    "primary_groups_inspect:", inspectPrimary.ndjson ?? "",
    "formula_error_scan_output:", errorText,
  ].join("\n");
  await fs.writeFile(VALIDATION_LOG, validationText + "\n", "utf8");
  await fs.rm(INSPECT_PATH, { force: true });
  await fs.rm(PAYLOAD_PATH, { force: true });
  console.log(`Workbook written: ${OUTPUT_PATH}`);
  console.log(`Workbook validation: ${hasErrors ? "FAIL" : "PASS"}`);
}

await main();
