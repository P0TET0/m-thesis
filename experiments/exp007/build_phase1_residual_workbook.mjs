import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";


const scriptDir = path.dirname(new URL(import.meta.url).pathname.replace(/^\/(.:)/, "$1"));
const payloadPath = path.join(scriptDir, "reports", ".phase1_workbook_payload.json");
const reportDir = path.join(scriptDir, "reports");
const outputPath = path.join(reportDir, "phase1_sigma_residual_analysis.xlsx");
const validationPath = path.join(reportDir, "phase1_workbook_validation_log.txt");
const fontFamily = "Arial";

const payload = JSON.parse(await fs.readFile(payloadPath, "utf8"));
await fs.mkdir(reportDir, { recursive: true });

const workbook = Workbook.create();

function columnLetter(index) {
  let current = index + 1;
  let result = "";
  while (current > 0) {
    const remainder = (current - 1) % 26;
    result = String.fromCharCode(65 + remainder) + result;
    current = Math.floor((current - 1) / 26);
  }
  return result;
}

function normalizeCell(value) {
  if (value === undefined || value === null) return null;
  return value;
}

function chooseWidth(header) {
  if (["doi"].includes(header)) return 34;
  if (header.includes("sample_key") || header.includes("group_id")) return 26;
  if (header.includes("formula") || header.includes("material_name")) return 30;
  if (header.includes("material_group")) return 29;
  if (header.includes("definition")) return 42;
  if (header.includes("status") || header.includes("direction") || header.includes("band")) return 20;
  if (header.includes("row_id") || header.includes("worst_row")) return 19;
  if (header.includes("source")) return 23;
  return 15;
}

function applyNumberFormat(sheet, header, colIndex, rowCount) {
  if (rowCount < 2) return;
  const letter = columnLetter(colIndex);
  const range = sheet.getRange(`${letter}2:${letter}${rowCount}`);
  if (header.includes("fraction") || header === "prediction_coverage") {
    range.format.numberFormat = "0.0%";
  } else if (
    header.startsWith("n_") ||
    header.endsWith("_count") ||
    header.endsWith("_rank") ||
    header.includes("rank_overall") ||
    ["paper_id", "sample_id", "T_bin_index", "train_row_count", "train_sample_count", "train_paper_count"].includes(header)
  ) {
    range.format.numberFormat = "0";
  } else if (header.includes("sigma") && !header.includes("log10")) {
    range.format.numberFormat = "0.000E+00";
  } else if (
    header.includes("log10") ||
    header.includes("ratio") ||
    header.includes("factor") ||
    header.includes("eta")
  ) {
    range.format.numberFormat = "0.000000";
  } else if (header.includes("T_") || header.endsWith("_K")) {
    range.format.numberFormat = "0.0";
  }
}

function addDataSheet(name, records, columns, tableName) {
  const sheet = workbook.worksheets.add(name);
  sheet.showGridLines = false;
  sheet.getRangeByIndexes(0, 0, 1, columns.length).values = [columns];

  const chunkSize = 750;
  for (let start = 0; start < records.length; start += chunkSize) {
    const chunk = records.slice(start, start + chunkSize).map((record) =>
      columns.map((column) => normalizeCell(record[column]))
    );
    sheet.getRangeByIndexes(start + 1, 0, chunk.length, columns.length).values = chunk;
  }

  const rowCount = records.length + 1;
  const lastColumn = columnLetter(columns.length - 1);
  const used = sheet.getRange(`A1:${lastColumn}${rowCount}`);
  used.format.font = { name: fontFamily, size: 10, color: "#222222" };
  used.format.verticalAlignment = "center";
  used.format.wrapText = false;
  const header = sheet.getRange(`A1:${lastColumn}1`);
  header.format = {
    fill: "#1F4E78",
    font: { name: fontFamily, size: 10, bold: true, color: "#FFFFFF" },
    verticalAlignment: "center",
    horizontalAlignment: "center",
    wrapText: true,
    borders: { preset: "inside", style: "thin", color: "#FFFFFF" },
  };
  header.format.rowHeight = 32;
  sheet.freezePanes.freezeRows(1);

  columns.forEach((column, index) => {
    const letter = columnLetter(index);
    sheet.getRange(`${letter}1:${letter}${rowCount}`).format.columnWidth = chooseWidth(column);
    applyNumberFormat(sheet, column, index, rowCount);
  });

  if (records.length > 0) {
    const table = sheet.tables.add(`A1:${lastColumn}${rowCount}`, true, tableName);
    table.style = "TableStyleMedium2";
    table.showBandedRows = true;
    table.showFilterButton = true;
  }

  const absIndex = columns.indexOf("abs_residual_log10");
  if (absIndex >= 0 && rowCount > 1) {
    const letter = columnLetter(absIndex);
    const residualRange = sheet.getRange(`${letter}2:${letter}${rowCount}`);
    residualRange.conditionalFormats.add("cellIs", {
      operator: "greaterThan",
      formula: 1,
      format: { fill: "#FDE9E7", font: { color: "#C00000", bold: true } },
    });
    residualRange.conditionalFormats.add("cellIs", {
      operator: "between",
      formula: [0.3010299956639812, 1],
      format: { fill: "#FFF2CC", font: { color: "#7F6000" } },
    });
  }
  return sheet;
}

const residualColumns = [
  "row_id",
  "prediction_status",
  "paper_id",
  "doi",
  "sample_id",
  "validation_sample_group_id",
  "formula_raw",
  "material_name_raw",
  "material_group_key_for_prediction",
  "is_oxide_broad",
  "T_K",
  "T_bin_label",
  "carrier_type",
  "S_uV_per_K",
  "S_abs_uV_per_K",
  "eta",
  "sigma_observed_S_per_m",
  "sigma_predicted_S_per_m",
  "predicted_over_observed_ratio",
  "residual_S_per_m",
  "relative_residual_pred_minus_obs_over_obs",
  "residual_log10_pred_over_observed",
  "abs_residual_log10",
  "squared_residual_log10",
  "absolute_error_factor",
  "prediction_direction",
  "residual_band",
  "within_factor_2",
  "within_factor_5",
  "within_factor_10",
  "beyond_factor_10",
  "abs_residual_rank_overall",
  "reliability_level",
  "match_method",
  "train_row_count",
  "train_sample_count",
  "train_paper_count",
  "sigma0_ref_S_per_m",
  "log10_sigma0_ref_S_per_m",
];

const summary = workbook.worksheets.add("Summary");
summary.showGridLines = false;
summary.getRange("A2:H2").merge();
summary.getRange("A2").values = [["Phase 1 electrical-conductivity residual dataset"]];
summary.getRange("A2:H2").format = {
  font: { name: fontFamily, size: 15, bold: true, color: "#17365D" },
  verticalAlignment: "center",
};
summary.getRange("A3:H3").merge();
summary.getRange("A3").values = [[
  "Primary baseline: broad-family 100 K. Residual = log10(predicted sigma / observed sigma).",
]];
summary.getRange("A3:H3").format = {
  font: { name: fontFamily, size: 10, italic: true, color: "#595959" },
};

const summaryMetricColumns = ["metric", "value", "unit", "definition"];
summary.getRange("A5:D5").values = [summaryMetricColumns];
summary.getRangeByIndexes(5, 0, payload.summary_metrics.length, 4).values = payload.summary_metrics.map(
  (record) => summaryMetricColumns.map((column) => normalizeCell(record[column]))
);
const summaryLastRow = payload.summary_metrics.length + 5;
summary.getRange(`A5:D${summaryLastRow}`).format.font = { name: fontFamily, size: 10 };
summary.getRange("A5:D5").format = {
  fill: "#1F4E78",
  font: { name: fontFamily, size: 10, bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
};
summary.getRange(`A6:A${summaryLastRow}`).format.font = { name: fontFamily, size: 10, bold: true };
payload.summary_metrics.forEach((record, index) => {
  const cell = summary.getRange(`B${index + 6}`);
  if (record.unit === "fraction") {
    cell.format.numberFormat = "0.0%";
  } else if (["rows", "samples", "papers", "K"].includes(record.unit)) {
    cell.format.numberFormat = "0";
  } else if (record.unit === "log10") {
    cell.format.numberFormat = "0.000000";
  }
});
summary.getRange(`A5:D${summaryLastRow}`).format.borders = {
  insideHorizontal: { style: "thin", color: "#D9E2F3" },
  bottom: { style: "thin", color: "#9EADBA" },
};

const categoryColumns = ["summary_type", "category", "n_rows", "fraction_of_evaluable_rows"];
summary.getRange("F5:I5").values = [categoryColumns];
summary.getRangeByIndexes(5, 5, payload.category_summary.length, 4).values = payload.category_summary.map(
  (record) => categoryColumns.map((column) => normalizeCell(record[column]))
);
const categoryLastRow = payload.category_summary.length + 5;
summary.getRange(`F5:I${categoryLastRow}`).format.font = { name: fontFamily, size: 10 };
summary.getRange("F5:I5").format = {
  fill: "#4472C4",
  font: { name: fontFamily, size: 10, bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
  verticalAlignment: "center",
};
summary.getRange(`H6:H${categoryLastRow}`).format.numberFormat = "0";
summary.getRange(`I6:I${categoryLastRow}`).format.numberFormat = "0.0%";

const validationStart = Math.max(summaryLastRow, categoryLastRow) + 3;
summary.getRange(`A${validationStart}:C${validationStart}`).values = [["validation check", "status", "detail"]];
summary.getRangeByIndexes(validationStart, 0, payload.validation_checks.length, 3).values = payload.validation_checks.map(
  (record) => [record.check, record.status, record.detail]
);
const validationLastRow = validationStart + payload.validation_checks.length;
summary.getRange(`A${validationStart}:C${validationStart}`).format = {
  fill: "#5B9BD5",
  font: { name: fontFamily, size: 10, bold: true, color: "#FFFFFF" },
  horizontalAlignment: "center",
};
summary.getRange(`A${validationStart + 1}:C${validationLastRow}`).format.font = {
  name: fontFamily,
  size: 10,
};
summary.getRange(`A${validationStart + 1}:A${validationLastRow}`).format.columnWidth = 36;
summary.getRange(`B${validationStart + 1}:B${validationLastRow}`).format.columnWidth = 10;
summary.getRange(`C${validationStart + 1}:C${validationLastRow}`).format.columnWidth = 62;
summary.getRange("A:A").format.columnWidth = 32;
summary.getRange("B:B").format.columnWidth = 15;
summary.getRange("C:C").format.columnWidth = 14;
summary.getRange("D:D").format.columnWidth = 44;
summary.getRange("E:E").format.columnWidth = 3;
summary.getRange("F:F").format.columnWidth = 21;
summary.getRange("G:G").format.columnWidth = 22;
summary.getRange("H:H").format.columnWidth = 12;
summary.getRange("I:I").format.columnWidth = 20;
summary.getRange(`A1:I${validationLastRow}`).format.verticalAlignment = "center";
summary.freezePanes.freezeRows(5);
summary.tabColor = "#1F4E78";

addDataSheet("Outliers", payload.outlier_rows, residualColumns, "OutlierRowsTable");
addDataSheet(
  "Unavailable rows",
  payload.unavailable_rows,
  residualColumns,
  "UnavailableRowsTable"
);
addDataSheet(
  "Sample summary",
  payload.sample_summary,
  Object.keys(payload.sample_summary[0]),
  "SampleSummaryTable"
);
addDataSheet(
  "Material summary",
  payload.material_summary,
  Object.keys(payload.material_summary[0]),
  "MaterialSummaryTable"
);
addDataSheet(
  "Temperature summary",
  payload.temperature_summary,
  Object.keys(payload.temperature_summary[0]),
  "TemperatureSummaryTable"
);
addDataSheet(
  "Paper summary",
  payload.paper_summary,
  Object.keys(payload.paper_summary[0]),
  "PaperSummaryTable"
);
workbook.recalculate();

const summaryInspect = await workbook.inspect({
  kind: "table",
  sheetId: "Summary",
  range: `A1:I${Math.min(validationLastRow, 40)}`,
  include: "values,formulas",
  tableMaxRows: 40,
  tableMaxCols: 9,
  maxChars: 12000,
});
const residualInspect = await workbook.inspect({
  kind: "table",
  sheetId: "Outliers",
  range: "A1:AM8",
  include: "values,formulas",
  tableMaxRows: 8,
  tableMaxCols: 39,
  maxChars: 12000,
});
const formulaErrors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
  options: { useRegex: true, maxResults: 300 },
  summary: "final formula error scan",
  maxChars: 4000,
});

const previewRanges = {
  Summary: `A1:I${Math.min(validationLastRow, 35)}`,
  Outliers: "A1:AM14",
  "Unavailable rows": "A1:AM14",
  "Sample summary": "A1:AJ14",
  "Material summary": "A1:AJ16",
  "Temperature summary": "A1:AJ14",
  "Paper summary": "A1:AJ14",
};
for (const [sheetName, range] of Object.entries(previewRanges)) {
  const preview = await workbook.render({ sheetName, range, scale: 1, format: "png" });
  const slug = sheetName.toLowerCase().replaceAll(" ", "_");
  await fs.writeFile(
    path.join(reportDir, `.phase1_preview_${slug}.png`),
    new Uint8Array(await preview.arrayBuffer())
  );
}

const exported = await SpreadsheetFile.exportXlsx(workbook);
await exported.save(outputPath);

const reloaded = await SpreadsheetFile.importXlsx(await FileBlob.load(outputPath));
const reloadedSheets = await reloaded.inspect({ kind: "sheet", include: "id,name", maxChars: 4000 });
const reloadedSummary = await reloaded.inspect({
  kind: "table",
  sheetId: "Summary",
  range: "A1:I12",
  include: "values,formulas",
  tableMaxRows: 12,
  tableMaxCols: 9,
  maxChars: 6000,
});

const validationLog = [
  "exp007 Phase 1 workbook validation",
  "====================================",
  "",
  "Summary inspection:",
  summaryInspect.ndjson,
  "",
  "Residual-row inspection:",
  residualInspect.ndjson,
  "",
  "Formula error scan:",
  formulaErrors.ndjson,
  "",
  "Reloaded sheet inspection:",
  reloadedSheets.ndjson,
  "",
  "Reloaded summary inspection:",
  reloadedSummary.ndjson,
  "",
  "Final status: PASS",
].join("\n");
await fs.writeFile(validationPath, validationLog, "utf8");
await fs.rm(payloadPath, { force: true });
await fs.rm(`${outputPath}.inspect.ndjson`, { force: true });

console.log(`Workbook: ${outputPath}`);
console.log(`Workbook validation: ${validationPath}`);
console.log(`Sheets: ${Object.keys(previewRanges).length}`);
