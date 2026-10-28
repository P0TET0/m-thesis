import fs from 'node:fs/promises';
import path from 'node:path';
import readline from 'node:readline';
import { createReadStream } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { Workbook } from '@oai/artifact-tool';

const base = path.dirname(fileURLToPath(import.meta.url));
if (process.argv.includes('--help-csv')) {
  const wb = Workbook.create();
  console.log(wb.help('workbook.toCSV', {include:'index,examples,notes', maxChars:5000}).ndjson);
} else {
  // Artifact Tool authors and validates each bounded table. Its documented
  // surface has no CSV exporter, so serialize its returned cell values as CSV.
  // Batching keeps the 104,832-row evidence-linked output within memory limits.
  const dryRun = process.argv.includes('--dry-run');
  const outputDir = path.join(base, dryRun ? 'cache/export_preview' : 'outputs');
  await fs.mkdir(outputDir, {recursive:true});
  const names = ['sample_host_review', 'paper_coverage', 'host_evidence'];
  const quote = value => {
    const s = value == null ? '' : String(value);
    return /[",\r\n\u2028\u2029]/.test(s) ? '"' + s.replaceAll('"', '""') + '"' : s;
  };
  const csvLine = row => row.map(quote).join(',') + '\r\n';
  const results = [];
  for (const name of names) {
    const input = path.join(base, 'cache', `${name}.jsonl`);
    const destination = path.join(outputDir, `${name}.csv`);
    const file = await fs.open(destination, 'w');
    await file.write('\uFEFF');
    const lines = readline.createInterface({input:createReadStream(input, {encoding:'utf8'}), crlfDelay:Infinity});
    let header, batch = [], count = 0, first = true;
    const flush = async () => {
      if (!batch.length) return;
      const wb = Workbook.create();
      const sheet = wb.worksheets.add('Review');
      const matrix = [header, ...batch];
      const range = sheet.getRangeByIndexes(0,0,matrix.length,header.length);
      range.values = matrix.map(row => row.map(value =>
        typeof value === 'string' && value.startsWith('=') ? "'" + value : value));
      wb.recalculate();
      const cells = range.values;
      if (JSON.stringify(cells) !== JSON.stringify(matrix)) {
        for (let r=0;r<matrix.length;r++) for (let c=0;c<header.length;c++) {
          if (JSON.stringify(cells[r]?.[c]) !== JSON.stringify(matrix[r][c])) {
            throw new Error(JSON.stringify({name,row:count+r,column:header[c],expected:matrix[r][c],actual:cells[r]?.[c]}));
          }
        }
      }
      if (first) {
        const inspect = await wb.inspect({kind:'table', range:'Review!A1:F4', include:'values', tableMaxRows:4,tableMaxCols:6,maxChars:1400});
        console.log(name, inspect.ndjson);
        await file.write(csvLine(cells[0]));
        first = false;
      }
      await file.write(cells.slice(1).map(csvLine).join(''));
      count += batch.length;
      batch = [];
    };
    for await (const line of lines) {
      if (!line) continue;
      const row = JSON.parse(line);
      header ??= Object.keys(row);
      batch.push(header.map(key=>row[key] ?? ''));
      if (batch.length >= 1000) await flush();
      if (dryRun && count >= 2000) break;
    }
    await flush();
    await file.close();
    results.push({name, rows:count, columns:header.length, path:destination});
    console.log(JSON.stringify(results.at(-1)));
  }
  await fs.writeFile(path.join(base,'cache','export_validation.json'),JSON.stringify(results,null,2));
}
