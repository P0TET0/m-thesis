import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";


function argsFrom(argv) {
  const result = {};
  for (let i = 0; i < argv.length; i += 2) {
    const key = argv[i];
    if (!key?.startsWith("--") || argv[i + 1] === undefined) {
      throw new Error(`Invalid argument sequence near: ${key ?? "<end>"}`);
    }
    result[key.slice(2)] = argv[i + 1];
  }
  return result;
}


function excelColumn(index) {
  let value = index + 1;
  let output = "";
  while (value > 0) {
    const remainder = (value - 1) % 26;
    output = String.fromCharCode(65 + remainder) + output;
    value = Math.floor((value - 1) / 26);
  }
  return output;
}


function parseCsvLine(line) {
  const values = [];
  let value = "";
  let quoted = false;
  for (let i = 0; i < line.length; i += 1) {
    const char = line[i];
    if (char === '"') {
      if (quoted && line[i + 1] === '"') {
        value += '"';
        i += 1;
      } else {
        quoted = !quoted;
      }
    } else if (char === "," && !quoted) {
      values.push(value);
      value = "";
    } else {
      value += char;
    }
  }
  values.push(value);
  return values;
}


function parseCsv(text) {
  const rows = [];
  let row = [];
  let value = "";
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    if (char === '"') {
      if (quoted && text[i + 1] === '"') {
        value += '"';
        i += 1;
      } else {
        quoted = !quoted;
      }
    } else if (char === "," && !quoted) {
      row.push(value);
      value = "";
    } else if ((char === "\n" || char === "\r") && !quoted) {
      if (char === "\r" && text[i + 1] === "\n") i += 1;
      row.push(value);
      value = "";
      if (row.some((cell) => cell !== "")) rows.push(row);
      row = [];
    } else {
      value += char;
    }
  }
  if (value !== "" || row.length) {
    row.push(value);
    rows.push(row);
  }
  return rows;
}


function typedCell(value, header) {
  if (value === "") return null;
  const name = header.toLowerCase();
  const textColumns = /key|^doi$|title|composition|material|source|reason|description|method|status|config|n_or_p|sample_id|row_id|flag|checked|metric|evaluation_level/i;
  if (!textColumns.test(name) && /^[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?$/.test(value)) {
    return Number(value);
  }
  if (/flag|checked|available|used/i.test(name) && /^(true|false)$/i.test(value)) {
    return value.toLowerCase() === "true";
  }
  return value;
}


function numericFormat(header) {
  const name = header.toLowerCase();
  if (name.includes("count") || name.includes("rows") || name.includes("points") || name.startsWith("number_of")) {
    return "#,##0";
  }
  if (name.includes("precision") || name.includes("recall") || name === "f1" || name.includes("correlation") || name === "value") {
    return "0.000";
  }
  if (name.includes("pct") || name.includes("mape")) {
    return "0.00";
  }
  if (name.includes("sigma") || name.includes("seebeck") || name.includes("kappa") || name.startsWith("pf_")) {
    return "0.000E+00";
  }
  if (name.includes("zt") || name.includes("error") || name.includes("rmse") || name.includes("mae")) {
    return "0.0000";
  }
  if (name.includes("temperature") || name.endsWith("_k")) {
    return "0.0";
  }
  return null;
}


const args = argsFrom(process.argv.slice(2));
if (args["verify-workbook"]) {
  const verifyPath = path.resolve(args["verify-workbook"]);
  const verifyPreviewDir = args["preview-dir"] ? path.resolve(args["preview-dir"]) : null;
  const blob = await FileBlob.load(verifyPath);
  const verifyWorkbook = await SpreadsheetFile.importXlsx(blob);
  const sheetInspect = await verifyWorkbook.inspect({ kind: "sheet", include: "id,name", maxChars: 5000 });
  console.log(sheetInspect.ndjson);
  const overviewCheck = await verifyWorkbook.inspect({
    kind: "table",
    range: "Overview!A1:H10",
    include: "values,formulas",
    tableMaxRows: 12,
    tableMaxCols: 8,
    maxChars: 6000,
  });
  console.log(overviewCheck.ndjson);
  const verifyErrors = await verifyWorkbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
    options: { useRegex: true, maxResults: 100 },
    summary: "imported workbook formula error scan",
    maxChars: 4000,
  });
  console.log(verifyErrors.ndjson);
  if (verifyPreviewDir) {
    await fs.mkdir(verifyPreviewDir, { recursive: true });
    const verifySheets = [
      "Overview",
      "Overall Metrics",
      "High ZT Classification",
      "Ranking Summary",
      "Sample Summary",
      "Row Predictions",
      "Problem Rows",
      "Problem Samples",
      "Notes",
    ];
    for (const sheetName of verifySheets) {
      const range = sheetName === "Overview" ? "A1:H10" : sheetName === "Notes" ? "A1:B11" : "A1:L25";
      const preview = await verifyWorkbook.render({ sheetName, range, scale: 1.2, format: "png" });
      await fs.writeFile(
        path.join(verifyPreviewDir, `${sheetName.replaceAll(" ", "_")}.png`),
        new Uint8Array(await preview.arrayBuffer()),
      );
    }
  }
  process.exit(0);
}
if (!args["input-dir"] || !args.output) {
  throw new Error("Usage: node build_exp006_pf_zt_workbook.mjs --input-dir DIR --output FILE [--preview-dir DIR]");
}

const inputDir = path.resolve(args["input-dir"]);
const outputPath = path.resolve(args.output);
const previewDir = args["preview-dir"] ? path.resolve(args["preview-dir"]) : null;

const csvSheets = [
  ["exp006_pf_zt_overall_metrics.csv", "Overall Metrics"],
  ["exp006_high_zt_classification.csv", "High ZT Classification"],
  ["exp006_ranking_summary.csv", "Ranking Summary"],
  ["exp006_pf_zt_sample_summary.csv", "Sample Summary"],
  ["exp006_pf_zt_row_predictions.csv", "Row Predictions"],
  ["exp006_problem_rows.csv", "Problem Rows"],
  ["exp006_problem_samples.csv", "Problem Samples"],
];

const workbook = Workbook.create();
const overview = workbook.worksheets.add("Overview");
overview.showGridLines = false;

const imported = [];
for (const [fileName, sheetName] of csvSheets) {
  const csvPath = path.join(inputDir, fileName);
  const csvText = await fs.readFile(csvPath, "utf8");
  const parsed = parseCsv(csvText.replace(/^\uFEFF/, ""));
  if (!parsed.length) throw new Error(`CSV has no rows: ${csvPath}`);
  const headers = parsed[0];
  const rows = parsed.map((sourceRow, rowIndex) => {
    const padded = Array.from({ length: headers.length }, (_, columnIndex) => sourceRow[columnIndex] ?? "");
    return rowIndex === 0 ? padded : padded.map((value, columnIndex) => typedCell(value, headers[columnIndex]));
  });
  const sheet = workbook.worksheets.add(sheetName);
  const chunkSize = 1500;
  for (let start = 0; start < rows.length; start += chunkSize) {
    const chunk = rows.slice(start, start + chunkSize);
    sheet.getRangeByIndexes(start, 0, chunk.length, headers.length).values = chunk;
  }
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(1);

  const physicalLineCount = Math.max(2, rows.length);
  const lastColumn = excelColumn(headers.length - 1);
  const headerRange = sheet.getRange(`A1:${lastColumn}1`);
  headerRange.format = {
    fill: "#0F5D66",
    font: { bold: true, color: "#FFFFFF", size: 10 },
    wrapText: true,
    verticalAlignment: "center",
    borders: { preset: "outside", style: "thin", color: "#0B4046" },
  };
  headerRange.format.rowHeight = 34;

  headers.forEach((header, columnIndex) => {
    const width = /source|reason|description|title|method|config|note/i.test(header)
      ? 28
      : /sample_key|composition|material_system/i.test(header)
        ? 22
        : 14;
    const columnRange = sheet.getRangeByIndexes(0, columnIndex, 2, 1);
    columnRange.format.columnWidth = width;
    const format = numericFormat(header);
    if (format) {
      sheet.getRange(`${excelColumn(columnIndex)}2:${excelColumn(columnIndex)}${physicalLineCount}`).format.numberFormat = format;
    }
  });
  imported.push({ sheetName, headers });
}

const metricCsv = await fs.readFile(path.join(inputDir, "exp006_pf_zt_overall_metrics.csv"), "utf8");
const metricLines = metricCsv.replace(/^\uFEFF/, "").split(/\r?\n/).filter(Boolean);
const metricHeaders = parseCsvLine(metricLines[0]);
const metricValues = parseCsvLine(metricLines[1]);
const metricColumns = new Map(metricHeaders.map((header, index) => [header, index]));

overview.getRange("A1:H1").merge();
overview.getRange("A1").values = [["exp006 PF/ZT downstream evaluation"]];
overview.getRange("A1:H1").format = {
  fill: "#123B4A",
  font: { bold: true, color: "#FFFFFF", size: 18 },
  verticalAlignment: "center",
};
overview.getRange("A1:H1").format.rowHeight = 34;
overview.getRange("A3:B3").values = [["Analysis scope", "Existing exp006 sigma prediction propagated with observed S and full-curve kappa"]];
overview.getRange("A4:B4").values = [["Temperature join", "Same-sample nearest kappa/ZT within ±2 K; no interpolation"]];
overview.getRange("A5:B5").values = [["Seebeck conversion", "S_uV_per_K × 1e-6 = V/K"]];
overview.getRange("A7:H7").values = [[
  "KPI",
  "Sigma log RMSE",
  "PF MAPE (%)",
  "ZT vs obs MAPE (%)",
  "ZT vs calc MAPE (%)",
  "PF rows",
  "ZT vs obs rows",
  "ZT vs calc rows",
]];
overview.getRange("A7:H7").format = {
  fill: "#DCECEF",
  font: { bold: true, color: "#123B4A" },
  wrapText: true,
  borders: { preset: "outside", style: "thin", color: "#9ABCC2" },
};
overview.getRange("A8:H8").values = [["Primary result", null, null, null, null, null, null, null]];
const overviewMetrics = [
  "sigma_log_RMSE",
  "PF_MAPE_pct",
  "ZT_pred_vs_ZT_obs_MAPE_pct",
  "ZT_pred_vs_ZT_calc_MAPE_pct",
  "valid_PF_rows",
  "valid_ZT_obs_rows",
  "valid_ZT_calc_rows",
];
overviewMetrics.forEach((metric, index) => {
  const sourceIndex = metricColumns.get(metric);
  if (sourceIndex === undefined) return;
  overview.getCell(7, index + 1).formulas = [[`='Overall Metrics'!${excelColumn(sourceIndex)}2`]];
});
overview.getRange("B8:E8").format.numberFormat = "0.000";
overview.getRange("F8:H8").format.numberFormat = "#,##0";
overview.getRange("A8:H8").format = {
  fill: "#F4F9FA",
  font: { size: 12 },
  borders: { preset: "outside", style: "thin", color: "#9ABCC2" },
};
overview.getRange("A10:H10").merge();
overview.getRange("A10").values = [[
  "Interpretation: this is a propagation study, not a fully predictive unknown-material ZT model, because S and kappa remain observed quantities.",
]];
overview.getRange("A10:H10").format = {
  fill: "#FFF4D6",
  font: { italic: true, color: "#6B4E00" },
  wrapText: true,
};
overview.getRange("A10:H10").format.rowHeight = 34;
overview.getRange("A3:A5").format.font = { bold: true, color: "#123B4A" };
overview.getRange("A1:H12").format.verticalAlignment = "center";
overview.getRange("A:A").format.columnWidth = 22;
overview.getRange("B:H").format.columnWidth = 18;
overview.freezePanes.freezeRows(1);

const notes = workbook.worksheets.add("Notes");
notes.showGridLines = false;
notes.getRange("A1:F1").merge();
notes.getRange("A1").values = [["Method and audit notes"]];
notes.getRange("A1:F1").format = {
  fill: "#123B4A",
  font: { bold: true, color: "#FFFFFF", size: 16 },
};
notes.getRange("A3:B11").values = [
  ["Item", "Definition"],
  ["PF prediction", "seebeck_obs_V_per_K^2 × sigma_pred_exp006_S_per_m"],
  ["ZT prediction", "PF_pred_exp006_W_per_mK2 × temperature_K / kappa_obs_W_per_mK"],
  ["Observed PF", "Same PF formula using sigma_obs_S_per_m"],
  ["Observation-derived ZT", "Same ZT formula using sigma_obs_S_per_m"],
  ["Join", "Original curves expanded and median-aggregated, then same-sample nearest kappa/ZT within ±2 K"],
  ["S unit", "V/K, converted from exp006 microV/K by 1e-6"],
  ["Invalid inputs", "Missing/nonpositive sigma_pred; missing S; or missing/nonpositive T/kappa as applicable"],
  ["Model scope", "No sigma recomputation, no ML retraining, no tau_eff refitting"],
];
notes.getRange("A3:B3").format = {
  fill: "#DCECEF",
  font: { bold: true, color: "#123B4A" },
};
notes.getRange("A3:B11").format.wrapText = true;
notes.getRange("A:A").format.columnWidth = 24;
notes.getRange("B:B").format.columnWidth = 72;
notes.getRange("A3:B11").format.rowHeight = 26;

const overviewInspect = await workbook.inspect({
  kind: "table",
  range: "Overview!A1:H10",
  include: "values,formulas",
  tableMaxRows: 12,
  tableMaxCols: 8,
  maxChars: 6000,
});
console.log(overviewInspect.ndjson);
const formulaErrors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
  maxChars: 4000,
});
console.log(formulaErrors.ndjson);

if (previewDir) {
  await fs.mkdir(previewDir, { recursive: true });
  const previewSheets = ["Overview", ...imported.map((item) => item.sheetName), "Notes"];
  for (const sheetName of previewSheets) {
    const sheet = workbook.worksheets.getItem(sheetName);
    const used = sheet.getUsedRange(true);
    const preview = await workbook.render({
      sheetName,
      range: sheetName === "Overview" ? "A1:H10" : sheetName === "Notes" ? "A1:B11" : `A1:${excelColumn(Math.min(imported.find((item) => item.sheetName === sheetName)?.headers.length ?? 12, 12) - 1)}25`,
      scale: 1.2,
      format: "png",
    });
    await fs.writeFile(
      path.join(previewDir, `${sheetName.replaceAll(" ", "_")}.png`),
      new Uint8Array(await preview.arrayBuffer()),
    );
    void used;
  }
}

await fs.mkdir(path.dirname(outputPath), { recursive: true });
const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(`Workbook exported: ${outputPath}`);
