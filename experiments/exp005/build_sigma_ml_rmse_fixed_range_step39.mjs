import fs from "node:fs/promises";
import os from "node:os";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const projectRoot = path.resolve(import.meta.dirname, "../..");
const sourcePath = path.join(
  projectRoot,
  "data/output/starrydata2_step32_sigma_ml_by_material_system",
  "sigma_ml_doi_holdout_predictions_by_material_system_step32.csv",
);
const outputDir = path.join(
  projectRoot,
  "data/output/starrydata2_step39_sigma_ml_rmse_fixed_1e2_1e8",
);
const outputPath = path.join(
  outputDir,
  "rmse_fixed_1e2_1e8_by_material_system_step39.xlsx",
);
const lower = 1e2;
const upper = 1e8;

const materials = [
  [1, "その他の化学式系", "broad::other_formula_system"],
  [2, "酸化物系", "broad::oxide"],
  [3, "Co-Sb系・スクッテルダイト系", "broad::CoSb_skutterudite_like"],
  [4, "セレン化物系", "broad::selenide"],
  [5, "Sb-Te系", "broad::SbTe_like"],
  [6, "Bi-Te系", "broad::BiTe_like"],
  [7, "Bi-Sb-Te系", "broad::BiSbTe_tetradymite_like"],
  [8, "硫化物系", "broad::sulfide"],
  [9, "Pb-Te系", "broad::PbTe_like"],
  [10, "Mg-Si/Sn系", "broad::Mg2SiSn_like"],
  [11, "テルル化物系", "broad::telluride"],
  [12, "Sn-Te系", "broad::SnTe_like"],
  [13, "Si-Ge系", "broad::SiGe_like"],
  [14, "Ge-Te系", "broad::GeTe_like"],
  [15, "分類不能", "unknown_material_group"],
  [16, "ハーフホイスラー系", "broad::half_heusler"],
];
const nameByKey = new Map(materials.map(([, name, key]) => [key, name]));

const csvText = await fs.readFile(sourcePath, "utf8");
const imported = await Workbook.fromCSV(csvText, { sheetName: "Source" });
const sourceSheet = imported.worksheets.getItem("Source");
const sourceValues = sourceSheet.getUsedRange(true).values;
const headers = sourceValues[0].map((value) => String(value));
const keyIndex = headers.indexOf("material_group_key_broad_family");
const expIndex = headers.indexOf("sigma_obs_S_per_m_step11");
const predIndex = headers.indexOf("sigma_pred_ML_S_per_m_step32");
if ([keyIndex, expIndex, predIndex].some((index) => index < 0)) {
  throw new Error("Source CSV is missing a required column");
}

const filteredRows = [];
for (const row of sourceValues.slice(1)) {
  const key = String(row[keyIndex] ?? "");
  const exp = Number(row[expIndex]);
  const pred = Number(row[predIndex]);
  if (
    Number.isFinite(exp) &&
    Number.isFinite(pred) &&
    exp >= lower &&
    exp <= upper &&
    pred >= lower &&
    pred <= upper
  ) {
    filteredRows.push([key, nameByKey.get(key) ?? key, exp, pred, null]);
  }
}
if (filteredRows.length === 0) {
  throw new Error("No observations remained after the fixed-range filter");
}

const workbook = Workbook.create();
const results = workbook.worksheets.add("Results");
const filtered = workbook.worksheets.add("Filtered data");
const method = workbook.worksheets.add("Method");

results.showGridLines = false;
results.getRange("A1:E1").merge();
results.getRange("A1").values = [["材料系別 RMSE（実測値・予測値ともに10^2～10^8 S/m）"]];
results.getRange("A2:B4").values = [
  ["下限 [S/m]", lower],
  ["上限 [S/m]", upper],
  ["RMSE定義", "sqrt(mean(log10(pred/exp)^2))"],
];
results.getRange("A6:E6").values = [[
  "順序",
  "材料系",
  "材料系キー",
  "範囲内点数",
  "RMSE [log10]",
]];
results.getRange("A7:C22").values = materials;
const dataLastRow = filteredRows.length + 1;
results.getRange("D7").formulas = [[
  `=COUNTIF('Filtered data'!$A$2:$A$${dataLastRow},C7)`,
]];
results.getRange("D7:D22").fillDown();
results.getRange("E7").formulas = [[
  `=IF(D7=0,"",SQRT(SUMIF('Filtered data'!$A$2:$A$${dataLastRow},C7,'Filtered data'!$E$2:$E$${dataLastRow})/D7))`,
]];
results.getRange("E7:E22").fillDown();
results.getRange("B24:C24").values = [["全材料系", "__all_material_systems__"]];
results.getRange("D24").formulas = [[`=ROWS('Filtered data'!$A$2:$A$${dataLastRow})`]];
results.getRange("E24").formulas = [[
  `=SQRT(AVERAGE('Filtered data'!$E$2:$E$${dataLastRow}))`,
]];

results.getRange("A1:E1").format = {
  fill: "#1B9E77",
  font: { bold: true, color: "#FFFFFF", size: 14 },
  verticalAlignment: "center",
};
results.getRange("A1:E1").format.rowHeight = 28;
results.getRange("A6:E6").format = {
  fill: "#D8EFE8",
  font: { bold: true, color: "#153E35" },
  borders: { preset: "doubleBottom", style: "thin", color: "#1B9E77" },
};
results.getRange("B24:E24").format = {
  fill: "#EAF6F2",
  font: { bold: true, color: "#153E35" },
  borders: { preset: "doubleBottom", style: "thin", color: "#1B9E77" },
};
results.getRange("D7:E22").format.fill = "#F2FAF7";
results.getRange("D7:D24").format.numberFormat = "#,##0";
results.getRange("E7:E24").format.numberFormat = "0.000000";
results.getRange("A2:A4").format.font = { bold: true, color: "#3F4F49" };
results.getRange("A1:E24").format.verticalAlignment = "center";
results.getRange("A1:E24").format.autofitColumns();
results.getRange("A:A").format.columnWidth = 14;
results.getRange("B:B").format.columnWidth = 29;
results.getRange("C:C").format.columnWidth = 38;
results.getRange("D:E").format.columnWidth = 18;
results.freezePanes.freezeRows(6);

filtered.showGridLines = false;
filtered.getRange("A1:E1").values = [[
  "材料系キー",
  "材料系",
  "実測σ [S/m]",
  "予測σ [S/m]",
  "log10(pred/exp)^2",
]];
filtered.getRange(`A2:E${dataLastRow}`).values = filteredRows;
filtered.getRange(`E2:E${dataLastRow}`).formulas = filteredRows.map((_, index) => {
  const row = index + 2;
  return [`=LOG10(D${row}/C${row})^2`];
});
filtered.getRange("A1:E1").format = {
  fill: "#1B9E77",
  font: { bold: true, color: "#FFFFFF" },
};
filtered.getRange(`C2:E${dataLastRow}`).format.numberFormat = "0.000000";
filtered.getRange(`A1:E${dataLastRow}`).format.autofitColumns();
filtered.getRange("A:A").format.columnWidth = 36;
filtered.getRange("B:B").format.columnWidth = 29;
filtered.getRange("C:E").format.columnWidth = 19;
filtered.freezePanes.freezeRows(1);
filtered.tables.add(`A1:E${dataLastRow}`, true, "FilteredSigmaData");

method.showGridLines = false;
method.getRange("A1:B1").values = [["項目", "内容"]];
method.getRange("A2:B8").values = [
  ["対象", "exp005 ML法の材料系別σ予測"],
  ["範囲条件", "実測σと予測σの両方が10^2以上10^8以下の点のみ（境界を含む）"],
  ["RMSE", "sqrt(mean(log10(pred/exp)^2))"],
  ["材料系", "exp006と同じbroad_family分類"],
  ["範囲内点数", filteredRows.length],
  ["元データ", sourcePath],
  ["計算方法", "Filtered dataの各点で二乗log誤差を計算し、Resultsで材料系別に集計"],
];
method.getRange("A1:B1").format = {
  fill: "#1B9E77",
  font: { bold: true, color: "#FFFFFF" },
};
method.getRange("A2:A8").format.font = { bold: true, color: "#3F4F49" };
method.getRange("A1:B8").format.wrapText = true;
method.getRange("A:A").format.columnWidth = 18;
method.getRange("B:B").format.columnWidth = 80;
method.getRange("A1:B8").format.autofitRows();
method.freezePanes.freezeRows(1);

const resultInspection = await workbook.inspect({
  kind: "table",
  range: "Results!A1:E24",
  include: "values,formulas",
  tableMaxRows: 30,
  tableMaxCols: 6,
});
console.log(resultInspection.ndjson);
const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "final formula error scan",
});
console.log(errors.ndjson);
if (/#[A-Z0-9?/]+!/.test(errors.ndjson)) {
  throw new Error("Formula error scan failed");
}

const verificationDir = path.join(os.tmpdir(), "exp005-step39-rmse-verification");
await fs.mkdir(verificationDir, { recursive: true });
for (const [sheetName, range, fileName] of [
  ["Results", "A1:E24", "results.png"],
  ["Filtered data", "A1:E20", "filtered-data.png"],
  ["Method", "A1:B8", "method.png"],
]) {
  const preview = await workbook.render({
    sheetName,
    range,
    scale: 1.4,
    format: "png",
  });
  await fs.writeFile(
    path.join(verificationDir, fileName),
    new Uint8Array(await preview.arrayBuffer()),
  );
}

await fs.mkdir(outputDir, { recursive: true });
const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(outputPath);
console.log(JSON.stringify({ outputPath, verificationDir, filteredRows: filteredRows.length }));
