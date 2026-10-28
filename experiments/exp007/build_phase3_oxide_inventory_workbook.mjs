import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";


const ROOT = process.cwd();
const EXP007 = path.join(ROOT, "experiments", "exp007");
const PAYLOAD_PATH = path.join(EXP007, "reports", ".phase3_workbook_payload.json");
const OUTPUT_PATH = path.join(EXP007, "reports", "phase3_oxide_inventory.xlsx");
const INSPECT_PATH = `${OUTPUT_PATH}.inspect.ndjson`;
const VALIDATION_LOG = path.join(EXP007, "reports", "phase3_workbook_validation_log.txt");
const PREVIEW_DIR = path.join(EXP007, "reports", ".phase3_workbook_previews");

const FONT = "Arial";
const COLORS = {
  navy: "#1F4E78",
  navy2: "#2F75B5",
  paleBlue: "#D9EAF7",
  paleGreen: "#E2F0D9",
  paleAmber: "#FFF2CC",
  paleRed: "#FCE4D6",
  redText: "#9C0006",
  amberText: "#9C6500",
  greenText: "#375623",
  gray: "#666666",
  lightGray: "#D9E2F3",
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
  const rows = records.map((record) => headers.map((header) => record[header] ?? null));
  return { headers, rows };
}

function applyBaseSheetStyle(sheet) {
  sheet.showGridLines = false;
  sheet.getRange("A1:AZ5000").format.font = { name: FONT, size: 10, color: "#222222" };
  sheet.getRange("A1:AZ5000").format.verticalAlignment = "center";
}

function styleTitle(sheet, title, note) {
  sheet.getRange("A2").values = [[title]];
  sheet.getRange("A2").format.font = { name: FONT, size: 15, bold: true, color: "#17365D" };
  sheet.getRange("A3:J3").format.fill = COLORS.navy;
  sheet.getRange("A3:J3").format.rowHeight = 3;
  sheet.getRange("A4").values = [[note]];
  sheet.getRange("A4").format.font = { name: FONT, size: 10, italic: true, color: COLORS.gray };
  sheet.getRange("A4").format.wrapText = false;
}

function widthForHeader(header) {
  const h = header.toLowerCase();
  if (h.includes("reason") || h.includes("rule") || h.includes("definition") || h.includes("detail")) return 38;
  if (h.includes("formula") || h.includes("material_name") || h.includes("material_label")) return 30;
  if (h.includes("path") || h.includes("doi")) return 30;
  if (h.includes("sha256")) return 28;
  if (h.includes("sample_key") || h.includes("validation_") || h === "row_id") return 21;
  if (h.includes("family") || h.includes("elements")) return 24;
  if (h.includes("priority") || h.includes("status") || h.includes("carrier")) return 14;
  if (h.startsWith("n_") || h.endsWith("_count") || h.includes("rank")) return 13;
  if (h.includes("fraction") || h.includes("share") || h.includes("accuracy")) return 15;
  if (h.includes("residual") || h.includes("rmse") || h.includes("mae")) return 16;
  if (h.includes("sigma") || h.includes("factor")) return 17;
  return Math.min(22, Math.max(12, header.length + 2));
}

function formatColumnByHeader(sheet, header, columnIndex, firstDataRow, lastDataRow) {
  if (lastDataRow < firstDataRow) return;
  const letter = columnLetter(columnIndex);
  const range = sheet.getRange(`${letter}${firstDataRow}:${letter}${lastDataRow}`);
  const h = header.toLowerCase();
  if (h.includes("fraction") || h.includes("share") || h.includes("accuracy")) {
    range.format.numberFormat = "0.0%";
  } else if (h.startsWith("n_") || h.endsWith("_count") || h.includes("rank") || h.endsWith("_id")) {
    range.format.numberFormat = "#,##0";
  } else if (h.includes("sigma_") && h.includes("per_m")) {
    range.format.numberFormat = "0.000E+00";
  } else if (h === "t_k" || h.includes("t_min") || h.includes("t_max") || h.includes("t_bin_center")) {
    range.format.numberFormat = "0.0";
  } else if (h.includes("residual") || h.includes("rmse") || h.includes("mae")) {
    range.format.numberFormat = "0.000";
  } else if (h.includes("factor")) {
    range.format.numberFormat = "0.00";
  }
}

function addPriorityFormatting(sheet, headers, firstDataRow, lastDataRow) {
  const index = headers.indexOf("review_priority");
  if (index < 0 || lastDataRow < firstDataRow) return;
  const letter = columnLetter(index);
  const range = sheet.getRange(`${letter}${firstDataRow}:${letter}${lastDataRow}`);
  range.conditionalFormats.add("containsText", {
    text: "high",
    format: { fill: COLORS.paleRed, font: { bold: true, color: COLORS.redText } },
  });
  range.conditionalFormats.add("containsText", {
    text: "medium",
    format: { fill: COLORS.paleAmber, font: { bold: true, color: COLORS.amberText } },
  });
  range.conditionalFormats.add("containsText", {
    text: "low",
    format: { fill: COLORS.paleBlue, font: { color: COLORS.navy } },
  });
}

function writeTableSheet(workbook, config) {
  const { name, title, note, records, tableName, tabColor = null } = config;
  const sheet = workbook.worksheets.add(name);
  applyBaseSheetStyle(sheet);
  styleTitle(sheet, title, note);
  if (tabColor) sheet.tabColor = tabColor;

  const { headers, rows } = matrixFromRecords(records);
  if (headers.length === 0) {
    sheet.getRange("A6").values = [["No records"]];
    return { sheet, headers, lastRow: 6, lastColumn: 0 };
  }
  const headerRow = 6;
  const firstDataRow = 7;
  const lastDataRow = firstDataRow + rows.length - 1;
  const lastColumn = headers.length - 1;
  const lastLetter = columnLetter(lastColumn);
  sheet.getRange(`A${headerRow}:${lastLetter}${headerRow}`).values = [headers];
  if (rows.length > 0) sheet.getRange(`A${firstDataRow}:${lastLetter}${lastDataRow}`).values = rows;

  const header = sheet.getRange(`A${headerRow}:${lastLetter}${headerRow}`);
  header.format.fill = COLORS.navy;
  header.format.font = { name: FONT, size: 10, bold: true, color: COLORS.white };
  header.format.horizontalAlignment = "center";
  header.format.verticalAlignment = "center";
  header.format.wrapText = true;
  header.format.rowHeight = 32;
  header.format.borders = { preset: "outside", style: "thin", color: COLORS.navy };

  if (rows.length > 0) {
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
    formatColumnByHeader(sheet, headerName, index, firstDataRow, lastDataRow);
    if (
      headerName.includes("reason") ||
      headerName.includes("rule") ||
      headerName.includes("formula") ||
      headerName.includes("material_name") ||
      headerName.includes("material_label")
    ) {
      sheet.getRange(`${letter}${firstDataRow}:${letter}${lastDataRow}`).format.wrapText = false;
    }
  });
  addPriorityFormatting(sheet, headers, firstDataRow, lastDataRow);

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
  return { sheet, headers, lastRow: Math.max(headerRow, lastDataRow), lastColumn };
}

function writeSummarySheet(workbook, payload) {
  const sheet = workbook.worksheets.add("Summary");
  applyBaseSheetStyle(sheet);
  styleTitle(sheet, payload.title, payload.scope_note);
  sheet.tabColor = COLORS.navy;
  sheet.getRange("A6:C6").values = [["metric", "value", "definition"]];
  const summary = payload.tables.summary;
  const rows = summary.map((record) => [record.metric, record.value, record.definition]);
  const lastRow = 6 + rows.length;
  sheet.getRange(`A7:C${lastRow}`).values = rows;
  sheet.getRange("A6:C6").format.fill = COLORS.navy;
  sheet.getRange("A6:C6").format.font = { name: FONT, size: 10, bold: true, color: COLORS.white };
  sheet.getRange("A6:C6").format.horizontalAlignment = "center";
  sheet.getRange(`A7:C${lastRow}`).format.borders = {
    insideHorizontal: { style: "thin", color: "#E7E6E6" },
    bottom: { style: "thin", color: "#B4C7E7" },
  };
  sheet.getRange("A:A").format.columnWidth = 35;
  sheet.getRange("B:B").format.columnWidth = 16;
  sheet.getRange("C:C").format.columnWidth = 55;
  summary.forEach((record, idx) => {
    const row = 7 + idx;
    if (String(record.metric).includes("fraction") || String(record.metric).includes("accuracy")) {
      sheet.getRange(`B${row}`).format.numberFormat = "0.0%";
    } else if (String(record.metric).includes("MAE")) {
      sheet.getRange(`B${row}`).format.numberFormat = "0.000";
    } else {
      sheet.getRange(`B${row}`).format.numberFormat = "#,##0";
    }
  });

  sheet.getRange("E6").values = [["Key findings"]];
  sheet.getRange("E6:H6").format.fill = COLORS.navy2;
  sheet.getRange("E6:H6").format.font = { name: FONT, size: 10, bold: true, color: COLORS.white };
  payload.insights.forEach((text, idx) => {
    sheet.getRange(`E${7 + idx}`).values = [[`${idx + 1}. ${text}`]];
  });
  sheet.getRange(`E7:E${6 + payload.insights.length}`).format.wrapText = true;
  sheet.getRange(`E7:E${6 + payload.insights.length}`).format.rowHeight = 34;
  sheet.getRange("E:E").format.columnWidth = 56;
  sheet.getRange("F:H").format.columnWidth = 3;

  sheet.getRange(`A${lastRow + 2}`).values = [["Interpretation limit"]];
  sheet.getRange(`A${lastRow + 2}:C${lastRow + 2}`).format.fill = COLORS.paleAmber;
  sheet.getRange(`A${lastRow + 2}`).format.font = { name: FONT, size: 10, bold: true, color: COLORS.amberText };
  sheet.getRange(`A${lastRow + 3}`).values = [[
    "Candidate families are formula-based audit groups. Structural classes such as perovskite or layered cobaltite require explicit rules or manual metadata."
  ]];
  sheet.getRange(`A${lastRow + 3}`).format.wrapText = false;
  sheet.getRange(`A${lastRow + 3}`).format.verticalAlignment = "top";
  sheet.getRange(`A${lastRow + 3}`).format.rowHeight = 22;
  sheet.freezePanes.freezeRows(6);
  return { sheet, lastRow: lastRow + 4, lastColumn: 7 };
}

function writeValidationSheet(workbook, payload) {
  const sheet = workbook.worksheets.add("Validation");
  applyBaseSheetStyle(sheet);
  styleTitle(
    sheet,
    "Validation and input manifest",
    "PASS/FAIL checks and SHA-256 hashes of the frozen source files used for this inventory."
  );
  sheet.tabColor = "#A5A5A5";

  const checks = matrixFromRecords(payload.tables.validation);
  const manifest = matrixFromRecords(payload.tables.manifest);
  sheet.getRange(`A6:${columnLetter(checks.headers.length - 1)}6`).values = [checks.headers];
  sheet.getRange(`A7:${columnLetter(checks.headers.length - 1)}${6 + checks.rows.length}`).values = checks.rows;
  sheet.getRange("F4").values = [["Input manifest"]];
  sheet.getRange("F4").format.font = { name: FONT, size: 11, bold: true, color: COLORS.navy };
  const manifestEnd = columnLetter(5 + manifest.headers.length - 1);
  sheet.getRange(`F6:${manifestEnd}6`).values = [manifest.headers];
  sheet.getRange(`F7:${manifestEnd}${6 + manifest.rows.length}`).values = manifest.rows;

  for (const rangeAddress of [
    `A6:${columnLetter(checks.headers.length - 1)}6`,
    `F6:${manifestEnd}6`,
  ]) {
    const header = sheet.getRange(rangeAddress);
    header.format.fill = COLORS.navy;
    header.format.font = { name: FONT, size: 10, bold: true, color: COLORS.white };
    header.format.horizontalAlignment = "center";
  }
  sheet.getRange(`B7:B${6 + checks.rows.length}`).conditionalFormats.add("containsText", {
    text: "FAIL",
    format: { fill: COLORS.paleRed, font: { bold: true, color: COLORS.redText } },
  });
  sheet.getRange("A:A").format.columnWidth = 33;
  sheet.getRange("B:B").format.columnWidth = 12;
  sheet.getRange("C:C").format.columnWidth = 45;
  sheet.getRange("F:F").format.columnWidth = 30;
  sheet.getRange("G:G").format.columnWidth = 48;
  sheet.getRange("H:H").format.columnWidth = 16;
  sheet.getRange("I:I").format.columnWidth = 28;
  sheet.freezePanes.freezeRows(6);
  return { sheet, lastRow: Math.max(6 + checks.rows.length, 6 + manifest.rows.length), lastColumn: 8 };
}

async function main() {
  const raw = await fs.readFile(PAYLOAD_PATH, "utf8");
  const payload = JSON.parse(raw);
  const workbook = Workbook.create();

  const built = [];
  built.push(writeSummarySheet(workbook, payload));
  built.push(writeTableSheet(workbook, {
    name: "Candidate families",
    title: "Provisional composition families",
    note: "Formula-based audit groups. Counts and residuals describe the held-out oxide cohort; they do not establish a final taxonomy.",
    records: payload.tables.families,
    tableName: "Phase3CandidateFamilies",
    tabColor: COLORS.navy2,
  }));
  built.push(writeTableSheet(workbook, {
    name: "Review queue",
    title: "Samples requiring chemistry review",
    note: "Prioritize likely organic/polymer false positives, then mixed-anion and free-text entries.",
    records: payload.tables.review,
    tableName: "Phase3ReviewQueue",
    tabColor: "#C55A11",
  }));
  built.push(writeTableSheet(workbook, {
    name: "Samples",
    title: "Broad-oxide sample inventory",
    note: "One row per validation sample group, with formula audit flags and residual summaries.",
    records: payload.tables.samples,
    tableName: "Phase3OxideSamples",
  }));
  built.push(writeTableSheet(workbook, {
    name: "Material labels",
    title: "Material-label inventory",
    note: "One row per normalized material label. Single-sample labels show where a label-level taxonomy would be sparse.",
    records: payload.tables.labels,
    tableName: "Phase3OxideLabels",
  }));
  built.push(writeTableSheet(workbook, {
    name: "Elements",
    title: "Element occurrence in the broad-oxide cohort",
    note: "Counts are non-exclusive because each material can contain several elements.",
    records: payload.tables.elements,
    tableName: "Phase3OxideElements",
  }));
  built.push(writeTableSheet(workbook, {
    name: "Temperature support",
    title: "Candidate-family support by carrier type and 100 K bin",
    note: "Evaluation-cohort counts only. The threshold mirrors 3 rows, 3 samples, and 1 paper but is not a training-feasibility result.",
    records: payload.tables.temperature,
    tableName: "Phase3TemperatureSupport",
  }));
  built.push(writeTableSheet(workbook, {
    name: "Oxide rows",
    title: "Row-level broad-oxide inventory",
    note: "Canonical row observations with current parsing evidence, provisional composition group, review flags, and Phase 1 residuals.",
    records: payload.tables.rows,
    tableName: "Phase3OxideRows",
  }));
  built.push(writeValidationSheet(workbook, payload));

  workbook.recalculate();
  await fs.mkdir(path.dirname(OUTPUT_PATH), { recursive: true });
  await fs.mkdir(PREVIEW_DIR, { recursive: true });

  const inspectSummary = await workbook.inspect({
    kind: "table",
    range: "Summary!A1:H25",
    include: "values,formulas",
    tableMaxRows: 25,
    tableMaxCols: 8,
    maxChars: 12000,
  });
  const inspectReview = await workbook.inspect({
    kind: "table",
    range: "Review queue!A1:N20",
    include: "values,formulas",
    tableMaxRows: 20,
    tableMaxCols: 14,
    maxChars: 12000,
  });
  const errors = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
    options: { useRegex: true, maxResults: 300 },
    summary: "final formula error scan",
    maxChars: 12000,
  });

  for (const entry of built) {
    const maxCol = Math.min(entry.lastColumn, 13);
    const maxRow = Math.min(entry.lastRow, 24);
    const range = `A1:${columnLetter(maxCol)}${maxRow}`;
    const preview = await workbook.render({
      sheetName: entry.sheet.name,
      range,
      scale: 1,
      format: "png",
    });
    const safeName = entry.sheet.name.replaceAll(" ", "_").toLowerCase();
    await fs.writeFile(
      path.join(PREVIEW_DIR, `${safeName}.png`),
      new Uint8Array(await preview.arrayBuffer())
    );
  }

  const output = await SpreadsheetFile.exportXlsx(workbook);
  await output.save(OUTPUT_PATH);
  const stat = await fs.stat(OUTPUT_PATH);
  const errorText = errors.ndjson ?? "";
  const hasFormulaErrors = /#REF!|#DIV\/0!|#VALUE!|#NAME\?|#N\/A|#NUM!|#NULL!|#SPILL!|#CALC!/.test(errorText);
  const validationText = [
    "Phase 3 workbook validation",
    `status=${hasFormulaErrors ? "FAIL" : "PASS"}`,
    `output=${path.relative(ROOT, OUTPUT_PATH).replaceAll("\\", "/")}`,
    `size_bytes=${stat.size}`,
    `sheets=${built.map((entry) => entry.sheet.name).join(" | ")}`,
    `formula_error_scan=${hasFormulaErrors ? "FAIL" : "PASS"}`,
    "summary_inspect:",
    inspectSummary.ndjson ?? "",
    "review_queue_inspect:",
    inspectReview.ndjson ?? "",
    "formula_error_scan_output:",
    errorText,
  ].join("\n");
  await fs.writeFile(VALIDATION_LOG, validationText + "\n", "utf8");
  await fs.rm(INSPECT_PATH, { force: true });
  await fs.rm(PAYLOAD_PATH, { force: true });
  console.log(`Workbook written: ${OUTPUT_PATH}`);
  console.log(`Workbook validation: ${hasFormulaErrors ? "FAIL" : "PASS"}`);
}

await main();
