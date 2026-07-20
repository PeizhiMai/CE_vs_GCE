import fs from 'node:fs/promises';
import path from 'node:path';
import { FileBlob, PresentationFile } from '@oai/artifact-tool';

// Import the validated 29-slide L=8/L=12 core, insert the 14-slide L=6
// section immediately before its cross-size conclusion, and update that
// conclusion to cover all three sizes.
const inputCore = process.argv[2];
const analysisDir = process.argv[3];
const outputPath = process.argv[4];
if (!inputCore || !analysisDir || !outputPath) {
  throw new Error('usage: node build_l6_43slide_deck.mjs input29.pptx l6_analysis_dir output43.pptx');
}

function parseTsv(text) {
  const lines = text.trim().split(/\r?\n/).filter(Boolean);
  const fields = lines.shift().split('\t');
  return lines.map((line) => {
    const values = line.split('\t'); const row = {};
    fields.forEach((field, i) => row[field] = values[i] ?? '');
    return row;
  });
}

const assets = parseTsv(await fs.readFile(path.join(analysisDir, 'l6_slide_assets.tsv'), 'utf8'))
  .sort((a, b) => Number(a.order) - Number(b.order));
if (assets.length !== 14) throw new Error(`expected 14 L=6 assets, found ${assets.length}`);
const summary = parseTsv(await fs.readFile(path.join(analysisDir, 'data', 'l6_thermometry_quantity_summary.tsv'), 'utf8'));
const pres = await PresentationFile.importPptx(await FileBlob.load(inputCore));
if (pres.slides.items.length !== 29) throw new Error(`expected 29-slide input core, found ${pres.slides.items.length}`);

const BG = '#FEFEFE', INK = '#182033', MUTED = '#657188', GRID = '#E6EAF0', ACCENT = '#D65B32';
function addText(slide, text, position, style = {}) {
  const box = slide.shapes.add({geometry:'textbox', position, fill:'none', line:{style:'solid', fill:'none', width:0}});
  box.text = text; box.text.style = {fontFamily:'Aptos', fontSize:16, color:INK, ...style}; return box;
}
function addRect(slide, position, fill = '#FFFFFF', lineFill = GRID, lineWidth = 1) {
  return slide.shapes.add({geometry:'rect', position, fill, line:{style:'solid', fill:lineFill, width:lineWidth}});
}
function slideHas(slide, needle) {
  return slide.elements.items.some((element) => element.text && element.text.toString().includes(needle));
}
function findSlideIndex(needle) {
  for (let i = 0; i < pres.slides.items.length; i++) if (slideHas(pres.slides.getItem(i), needle)) return i;
  return -1;
}
function pct(value) {
  const x = Number(value); return Number.isFinite(x) ? `${(100*x).toFixed(1)}%` : '—';
}
function summaryRow(key) { return summary.find((row) => row.quantity === key); }
function subtitleFor(row) {
  if (row.kind === 'thermometer') return 'Bias is (T_GCE−T_CE)/T_CE. Lines join actual simulated unique-root points only and break at no-T, ambiguous, flat, sign-limited, or unfinished conditions.';
  if (row.kind === 'thermometry_scorecard') return 'Invertibility and coverage are reported alongside median |(T_GCE−T_CE)/T_CE|; unresolved conditions are never assigned a temperature.';
  if (row.kind === 'mismatch') return 'Matched CE/GCE points use actual T=1/β. Ratios are dimensionless and are not multiplied by 100.';
  return 'CE versus GCE at actual T=1/β. Positive-U canonical rows use spin-HS, phase reweighting, and the corrected global phase-sum pool.';
}

const conclusionNeedles = [
  'Kinetic energy is the only thermometer that is robust at both sizes',
  'Kinetic energy is the most robust thermometer',
];
let conclusionIndex = conclusionNeedles.map(findSlideIndex).find((x) => x >= 0);
if (!(conclusionIndex >= 0)) conclusionIndex = pres.slides.items.length - 1;
let insertAfter = conclusionIndex - 1;

for (const row of assets) {
  const slide = pres.slides.insert({after: insertAfter}).slide;
  insertAfter += 1;
  slide.background.fill = BG;
  const titleSize = row.title.length > 58 ? 23.5 : 27;
  addText(slide, row.title, {left:42, top:18, width:1120, height:42}, {fontSize:titleSize, bold:true});
  addText(slide, subtitleFor(row), {left:43, top:58, width:1120, height:25}, {fontSize:11.8, color:MUTED});
  addText(slide, '0/43', {left:1190, top:25, width:66, height:20}, {fontSize:12.5, color:ACCENT, alignment:'right'});
  const png = await fs.readFile(row.png);
  const isThermometer = row.kind === 'thermometer';
  const top = isThermometer ? 82 : 88;
  const height = isThermometer ? 588 : 575;
  slide.images.add({blob:new Uint8Array(png), contentType:'image/png', alt:row.title,
                    fit:'contain', position:{left:10, top, width:1260, height}});
  if (isThermometer) {
    const stat = summaryRow(row.quantity);
    const text = stat ? `Unique T ${pct(stat.unique_fraction)}   •   No inferred T ${pct(stat.no_solution_fraction)}   •   Ambiguous/flat ${pct(stat.ambiguous_or_flat_fraction)}   •   Sign-limited ${pct(stat.sign_limited_fraction)}   •   Median |ΔT|/T_CE ${pct(stat.median_abs_deltaT_over_TCE)}` : '';
    addRect(slide, {left:60, top:654, width:1160, height:34}, '#F5F7FA', '#D9E0E8', 1);
    addText(slide, text, {left:72, top:661, width:1136, height:19}, {fontSize:11.5, bold:true, alignment:'center'});
  }
  addText(slide, 'Source: strict-final L=6 CE/GCE production; U=0 exact finite 6×6 PBC. Density-tuned GCE rows require |N−Ntarget|≤0.03.',
          {left:43, top:699, width:1180, height:14}, {fontSize:8.7, color:MUTED});
}

if (pres.slides.items.length !== 43) throw new Error(`expected 43 slides after insertion, found ${pres.slides.items.length}`);

// Update the pre-existing cross-size conclusion in place.  Its original L=8
// and L=12 chart remains useful context; the native L=6 callout below adds the
// third-size result without rasterizing the slide.
const conclusion = pres.slides.getItem(pres.slides.items.length - 1);
for (const element of conclusion.elements.items) {
  if (!element.text) continue;
  const text = element.text.toString();
  if (text.includes('Kinetic energy is the only thermometer that is robust at both sizes')) {
    element.text.replace(text, 'Kinetic energy is the most robust thermometer across L=6, L=8, and L=12');
  } else if (text.includes('Cross-size summary of unique-root bias')) {
    element.text.replace(text, 'Cross-size conclusion using unique-root bias, no-solution coverage, ambiguous calibration, and sign-limited coverage.');
  } else if (text.includes('Recommendation: use kinetic energy for the main')) {
    element.text.replace(text, 'Recommendation: use kinetic energy for the primary CE↔GCE thermometry claim across L=6,8,12; retain charge as a corroborating observable only where its calibration is unique.');
  } else if (text.includes('Source: refreshed L=8 and L=12 thermometry summary')) {
    element.text.replace(text, 'Source: refreshed L=6, L=8, and L=12 equal-time thermometry; every bias uses (T_GCE−T_CE)/T_CE and unique roots only.');
  }
}
const kinetic = summaryRow('kinetic');
if (kinetic) {
  addRect(conclusion, {left:55, top:577, width:1170, height:21}, '#FFF4EE', '#F1C7B7', 1);
  addText(conclusion,
    `New L=6 result — kinetic: unique T ${pct(kinetic.unique_fraction)}, no inferred T ${pct(kinetic.no_solution_fraction)}, median |ΔT|/T_CE ${pct(kinetic.median_abs_deltaT_over_TCE)}.`,
    {left:68, top:579, width:1144, height:17}, {fontSize:10.7, bold:true, color:ACCENT, alignment:'center'});
}

for (let i = 0; i < pres.slides.items.length; i++) {
  const slide = pres.slides.getItem(i);
  for (const element of slide.elements.items) {
    if (!element.text) continue;
    const text = element.text.toString().trim();
    if (/^\d+\/(25|29|32|43)$/.test(text)) element.text.replace(text, `${i+1}/43`);
  }
}

const inspection = await pres.inspect({kind:'slide,textbox,shape,image,table,chart,layout', maxChars:500000});
await fs.writeFile(`${outputPath}.inspect.ndjson`, inspection.ndjson);
const output = await PresentationFile.exportPptx(pres);
await output.save(outputPath);
console.log(outputPath);
