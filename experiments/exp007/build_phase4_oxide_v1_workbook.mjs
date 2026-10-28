import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";


const ROOT = process.cwd();
const EXP007 = path.join(ROOT, "experiments", "exp007");
const PAYLOAD_PATH = path.join(EXP007, "reports", ".phase4_workbook_payload.json");
const OUTPUT_PATH = path.join(EXP007, "reports", "phase4_oxide_v1_definition.xlsx");
const INSPECT_PATH = `${OUTPUT_PATH}.inspect.ndjson`;
const VALIDATION_LOG = path.join(EXP007, "reports", "phase4_workbook_validation_log.txt");
const PREVIEW_DIR = path.join(EXP007, "reports", ".phase4_workbook_previews");

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
  if (h.includes("description") || h.includes("reason") || h.includes("condition") || h.includes("definition") || h.includes("detail")) return 38;
  if (h.includes("group_key") || h.includes("display_name")) return 30;
  if (h.includes("formula") || h.includes("material_name") || h.includes("material_label")) return 30;
  if (h.includes("path") || h.includes("doi")) return 28;
  if (h.includes("sha256")) return 24;
  if (h.includes("sample_key") || h.includes("validation_") || h === "row_id") return 20;
  if (h.includes("scope") || h.includes("action") || h.includes("elements")) return 23;
  if (h.includes("priority") || h.includes("status") || h.includes("confidence")) return 15;
  if (h.startsWith("n_") || h.endsWith("_count") || h.includes("rank") || h === "priority") return 13;
  if (h.includes("fraction") || h.includes("share") || h.includes("accuracy")) return 15;
  if (h.includes("residual") || h.includes("rmse") || h.includes("mae")) return 16;
  return Math.min(22, Math.max(12, header.length + 2));
}

function formatColumn(sheet, header, index, firstDataRow, lastDataRow) {
  if (lastDataRow < firstDataRow) return;
  const letter = columnLetter(index);
  const range = sheet.getRange(`${letter}${firstDataRow}:${letter}${lastDataRow}`);
  const h = header.toLowerCase();
  if (h.includes("fraction") || h.includes("share") || h.includes("accuracy")) {
    range.format.numberFormat = "0.0%";
  } else if (h.startsWith("n_") || h.endsWith("_count") || h.includes("rank") || h === "priority") {
    range.format.numberFormat = "#,##0";
  } else if (h.includes("sigma_") && h.includes("per_m")) {
    range.format.numberFormat = "0.000E+00";
  } else if (h === "t_k" || h.includes("t_min") || h.includes("t_max")) {
    range.format.numberFormat = "0.0";
  } else if (h.includes("residual") || h.includes("rmse") || h.includes("mae")) {
    range.format.numberFormat = "0.000";
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
  const priorityIndex = headers.indexOf("oxide_v1_review_priority");
  if (priorityIndex >= 0) {
    const letter = columnLetter(priorityIndex);
    const range = sheet.getRange(`${letter}${firstDataRow}:${letter}${lastDataRow}`);
    addTextFormat(range, "high", COLORS.paleRed, COLORS.redText);
    addTextFormat(range, "medium", COLORS.paleAmber, COLORS.amberText);
    addTextFormat(range, "low", COLORS.paleBlue, COLORS.navy);
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

  const maeIndex = headers.indexOf("mae_log10");
  if (maeIndex >= 0 && rows.length > 1) {
    const letter = columnLetter(maeIndex);
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

  sheet.getRange("A6:C6").values = [["metric", "value", "definition"]];
  const rows = payload.tables.summary.map((record) => [record.metric, record.value, record.definition]);
  const lastRow = 6 + rows.length;
  sheet.getRange(`A7:C${lastRow}`).values = rows;
  sheet.getRange("A6:C6").format.fill = COLORS.navy;
  sheet.getRange("A6:C6").format.font = { name: FONT, size: 10, bold: true, color: COLORS.white };
  sheet.getRange(`A7:C${lastRow}`).format.borders = {
    insideHorizontal: { style: "thin", color: "#E7E6E6" },
    bottom: { style: "thin", color: "#B4C7E7" },
  };
  sheet.getRange("A:A").format.columnWidth = 34;
  sheet.getRange("B:B").format.columnWidth = 44;
  sheet.getRange("C:C").format.columnWidth = 56;
  payload.tables.summary.forEach((record, idx) => {
    const metric = String(record.metric);
    const row = 7 + idx;
    if (metric.includes("fraction") || metric.includes("accuracy")) sheet.getRange(`B${row}`).format.numberFormat = "0.0%";
    else if (metric.includes("MAE")) sheet.getRange(`B${row}`).format.numberFormat = "0.000";
    else if (!metric.includes("version") && !metric.includes("date") && !metric.includes("sha256")) sheet.getRange(`B${row}`).format.numberFormat = "#,##0";
    if (metric.includes("sha256")) {
      sheet.getRange(`B${row}`).format.font = { name: FONT, size: 8, color: "#222222" };
      sheet.getRange(`B${row}`).format.wrapText = true;
      sheet.getRange(`B${row}`).format.rowHeight = 30;
    }
  });

  sheet.getRange("E6:H6").format.fill = COLORS.blue;
  sheet.getRange("E6").values = [["Key findings"]];
  sheet.getRange("E6:H6").format.font = { name: FONT, size: 10, bold: true, color: COLORS.white };
  payload.insights.forEach((text, idx) => sheet.getRange(`E${7 + idx}`).values = [[`${idx + 1}. ${text}`]]);
  sheet.getRange(`E7:E${6 + payload.insights.length}`).format.wrapText = true;
  sheet.getRange(`E7:E${6 + payload.insights.length}`).format.rowHeight = 38;
  sheet.getRange("E:E").format.columnWidth = 62;
  sheet.getRange("F:H").format.columnWidth = 3;

  const limitRow = lastRow + 2;
  sheet.getRange(`A${limitRow}:C${limitRow}`).format.fill = COLORS.paleAmber;
  sheet.getRange(`A${limitRow}`).values = [["Interpretation limit"]];
  sheet.getRange(`A${limitRow}`).format.font = { name: FONT, size: 10, bold: true, color: COLORS.amberText };
  sheet.getRange(`A${limitRow + 1}`).values = [["oxide_v1 is composition-only. Perovskite, spinel, delafossite, rocksalt, and layered-cobaltite labels are intentionally not inferred from formula text."]];
  sheet.getRange(`A${limitRow + 1}:C${limitRow + 1}`).format.fill = "#FFF9E6";
  sheet.getRange(`A${limitRow + 1}`).format.wrapText = true;
  sheet.getRange(`A${limitRow + 1}`).format.rowHeight = 34;
  sheet.freezePanes.freezeRows(6);
  return { sheet, lastRow: limitRow + 1, lastColumn: 7 };
}

function writeValidationSheet(workbook, payload) {
  const sheet = workbook.worksheets.add("Validation");
  applyBaseStyle(sheet);
  styleTitle(sheet, "Validation and input manifest", "Build checks and SHA-256 hashes for the frozen Phase 3 inputs.");
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
    const header = sheet.getRange(address);
    header.format.fill = COLORS.navy;
    header.format.font = { name: FONT, size: 10, bold: true, color: COLORS.white };
  }
  addTextFormat(sheet.getRange(`B7:B${6 + checks.rows.length}`), "PASS", COLORS.paleGreen, COLORS.greenText);
  addTextFormat(sheet.getRange(`B7:B${6 + checks.rows.length}`), "FAIL", COLORS.paleRed, COLORS.redText);
  sheet.getRange("A:A").format.columnWidth = 34;
  sheet.getRange("B:B").format.columnWidth = 12;
  sheet.getRange("C:C").format.columnWidth = 48;
  sheet.getRange("F:F").format.columnWidth = 28;
  sheet.getRange("G:G").format.columnWidth = 52;
  sheet.getRange("H:H").format.columnWidth = 16;
  sheet.getRange("I:I").format.columnWidth = 28;
  sheet.freezePanes.freezeRows(6);
  return { sheet, lastRow: Math.max(6 + checks.rows.length, 6 + manifest.rows.length), lastColumn: 8 };
}

async function main() {
  const payload = JSON.parse(await fs.readFile(PAYLOAD_PATH, "utf8"));
  const workbook = Workbook.create();
  const built = [];
  built.push(writeSummarySheet(workbook, payload));
  const rules = reorderRecords(payload.tables.rules, ["priority", "rule_id", "display_name_ja", "group_key", "taxonomy_scope", "default_model_action", "required_elements_all", "required_elements_any", "label_condition", "matched_samples", "matched_rows", "matched_papers", "matched_material_labels", "rule_description"]);
  const classes = reorderRecords(payload.tables.classes, ["oxide_v1_display_name_ja", "oxide_v1_group_key", "oxide_v1_rule_id", "oxide_v1_default_model_action", "n_samples", "n_rows", "n_papers", "n_material_labels", "n_default_eligible_rows", "n_review_rows", "n_excluded_rows", "mae_log10", "rmse_log10", "within_factor_10_fraction"]);
  const samples = reorderRecords(payload.tables.samples, ["oxide_v1_inclusion_status", "oxide_v1_review_priority", "oxide_v1_review_reasons", "oxide_v1_display_name_ja", "oxide_v1_group_key", "formula_raw", "material_name_raw", "parsed_elements", "oxide_v1_default_model_eligible", "oxide_v1_formula_like_label", "oxide_v1_multicomponent_notation", "validation_sample_group_id", "doi", "n_rows", "median_residual_log10", "mae_log10", "within_factor_10_fraction"]);
  built.push(writeTableSheet(workbook, { name: "Rules", title: "Ordered oxide_v1 classification rules", note: "Rules are evaluated from lowest priority number to highest; the first match wins.", records: rules, tableName: "Phase4Rules", tabColor: COLORS.blue }));
  built.push(writeTableSheet(workbook, { name: "Class summary", title: "oxide_v1 class coverage and current residuals", note: "Performance columns describe the existing broad-family predictions before taxonomy-specific retraining.", records: classes, tableName: "Phase4Classes", tabColor: COLORS.blue }));
  built.push(writeTableSheet(workbook, { name: "Review queue", title: "Review and exclusion queue", note: "57 samples require confirmation: excluded organic/composite entries first, then medium- and low-priority review items.", records: payload.tables.review, tableName: "Phase4Review", tabColor: "#C55A11" }));
  built.push(writeTableSheet(workbook, { name: "Sample assignments", title: "One oxide_v1 assignment per validation sample", note: "Canonical sample-level classification used to prevent within-sample assignment conflicts.", records: samples, tableName: "Phase4Samples" }));
  built.push(writeTableSheet(workbook, { name: "Label assignments", title: "Material-label assignments", note: "One row per normalized material label, including default eligibility and review reasons.", records: payload.tables.labels, tableName: "Phase4Labels" }));
  built.push(writeTableSheet(workbook, { name: "Phase3 transition", title: "Transition from Phase 3 candidate groups", note: "Crosswalk between provisional Phase 3 audit groups and the frozen oxide_v1 definition.", records: payload.tables.transition, tableName: "Phase4Transition" }));
  built.push(writeTableSheet(workbook, { name: "Rule examples", title: "Representative labels for each rule", note: "Examples support manual review of rule boundaries; they are not additional classification logic.", records: payload.tables.examples, tableName: "Phase4Examples" }));
  built.push(writeTableSheet(workbook, { name: "Row assignments", title: "Row-level oxide_v1 assignments", note: "All 3,437 current broad-oxide held-out rows with Phase 3 evidence and frozen Phase 4 assignments.", records: payload.tables.rows, tableName: "Phase4Rows" }));
  built.push(writeValidationSheet(workbook, payload));

  workbook.recalculate();
  await fs.mkdir(path.dirname(OUTPUT_PATH), { recursive: true });
  await fs.mkdir(PREVIEW_DIR, { recursive: true });
  const inspectSummary = await workbook.inspect({ kind: "table", range: "Summary!A1:H28", include: "values,formulas", tableMaxRows: 28, tableMaxCols: 8, maxChars: 12000 });
  const inspectReview = await workbook.inspect({ kind: "table", range: "Review queue!A1:P18", include: "values,formulas", tableMaxRows: 18, tableMaxCols: 16, maxChars: 12000 });
  const errors = await workbook.inspect({ kind: "match", searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!", options: { useRegex: true, maxResults: 300 }, summary: "final formula error scan", maxChars: 12000 });

  for (const entry of built) {
    const range = `A1:${columnLetter(Math.min(entry.lastColumn, 13))}${Math.min(entry.lastRow, 24)}`;
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
    "Phase 4 workbook validation",
    `status=${hasErrors ? "FAIL" : "PASS"}`,
    `output=${path.relative(ROOT, OUTPUT_PATH).replaceAll("\\", "/")}`,
    `size_bytes=${stat.size}`,
    `sheets=${built.map((entry) => entry.sheet.name).join(" | ")}`,
    `formula_error_scan=${hasErrors ? "FAIL" : "PASS"}`,
    "summary_inspect:", inspectSummary.ndjson ?? "",
    "review_queue_inspect:", inspectReview.ndjson ?? "",
    "formula_error_scan_output:", errorText,
  ].join("\n");
  await fs.writeFile(VALIDATION_LOG, validationText + "\n", "utf8");
  await fs.rm(INSPECT_PATH, { force: true });
  await fs.rm(PAYLOAD_PATH, { force: true });
  console.log(`Workbook written: ${OUTPUT_PATH}`);
  console.log(`Workbook validation: ${hasErrors ? "FAIL" : "PASS"}`);
}

await main();
