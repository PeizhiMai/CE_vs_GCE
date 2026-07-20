import fs from 'node:fs/promises';
import path from 'node:path';
import { Presentation, PresentationFile } from '@oai/artifact-tool';

// Standalone 18-slide L=6 OBC deck.  The 14 scientific panels are supplied
// by analyze_l6_obc_thermometry.py; all framing, comparison charts, scorecards,
// conclusions, and provenance remain editable PowerPoint objects.
const analysisDir = process.argv[2];
const outputPath = process.argv[3];
const pbcSummaryPath = process.argv[4];
const pbcSnapshotPath = process.argv[5];
const pbcStatusPath = process.argv[6];
if (!analysisDir || !outputPath || !pbcSummaryPath || !pbcSnapshotPath || !pbcStatusPath) {
  throw new Error('usage: node build_l6_obc_18slide_deck.mjs ANALYSIS_DIR OUTPUT.pptx PBC_SUMMARY.tsv PBC_SNAPSHOT.tsv PBC_STATUS.tsv');
}

function parseTsv(text) {
  const lines = text.replace(/^\uFEFF/, '').trim().split(/\r?\n/).filter(Boolean);
  if (!lines.length) return [];
  const fields = lines.shift().split('\t');
  return lines.map((line) => {
    const values = line.split('\t');
    const row = {};
    fields.forEach((field, i) => { row[field] = values[i] ?? ''; });
    return row;
  });
}

const readTsv = async (file) => parseTsv(await fs.readFile(file, 'utf8'));
const assets = (await readTsv(path.join(analysisDir, 'l6_slide_assets.tsv')))
  .sort((a, b) => Number(a.order) - Number(b.order));
if (assets.length !== 14) throw new Error(`expected 14 OBC analysis assets, found ${assets.length}`);
const obcSummary = await readTsv(path.join(analysisDir, 'data', 'l6_thermometry_quantity_summary.tsv'));
const obcScoreRows = await readTsv(path.join(analysisDir, 'data', 'l6_obc_workflow_scorecard.tsv'));
const pbcSummary = await readTsv(pbcSummaryPath);
const pbcSnapshot = await readTsv(pbcSnapshotPath);
const pbcStatus = await readTsv(pbcStatusPath);
if (obcSummary.length !== 4 || pbcSummary.length !== 4 || obcScoreRows.length !== 1) {
  throw new Error(`expected four PBC/four OBC thermometer summaries and one OBC workflow score row`);
}

const QORDER = ['kinetic', 'double_occupancy', 'nn_spin', 'nn_charge_connected'];
const QSHORT = {
  kinetic: 'Kinetic',
  double_occupancy: 'Double occ.',
  nn_spin: 'NN spin',
  nn_charge_connected: 'NN charge',
};
const qmap = (rows) => new Map(rows.map((row) => [row.quantity, row]));
const OBC = qmap(obcSummary);
const PBC = qmap(pbcSummary);
for (const key of QORDER) if (!OBC.has(key) || !PBC.has(key)) throw new Error(`missing thermometry summary for ${key}`);
const S = obcScoreRows[0];

function num(value, fallback = NaN) {
  const x = Number(value);
  return Number.isFinite(x) ? x : fallback;
}
function pct(value, digits = 1) {
  const x = num(value);
  return Number.isFinite(x) ? `${(100 * x).toFixed(digits)}%` : '—';
}
function fixed(value, digits = 3) {
  const x = num(value);
  return Number.isFinite(x) ? x.toFixed(digits) : '—';
}
function countPbcStatus(predicate) {
  return pbcStatus.filter(predicate).length;
}
const isPositivePilot = (row) => num(row.U) > 0 && [5, 6.7, 10].some((b) => Math.abs(num(row.beta) - b) < 1e-9) && row.ensemble === 'CE';
const pbcPilotRows = pbcStatus.filter(isPositivePilot);
const pbcPilotAdmitted = pbcPilotRows.filter((r) => r.status === 'strict_final').length;
const pbcPilotLimited = pbcPilotRows.filter((r) => r.status === 'sign_limited').length;
const pbcGce = pbcSnapshot.filter((r) => r.ensemble === 'GCE' && Math.abs(num(r.U)) > 1e-12);
const median = (values) => {
  const x = values.filter(Number.isFinite).sort((a, b) => a - b);
  if (!x.length) return NaN;
  const i = Math.floor(x.length / 2);
  return x.length % 2 ? x[i] : (x[i - 1] + x[i]) / 2;
};
const pbcDensityErrors = pbcGce.map((r) => Math.abs(num(r.N_mean) - num(r.Ntot)));
const pbcMuShifts = pbcGce.map((r) => Math.abs(num(r.mu) - num(r.mu_L8_reference)));
const pbcDensityMedian = median(pbcDensityErrors);
const pbcDensityMax = pbcDensityErrors.length ? Math.max(...pbcDensityErrors.filter(Number.isFinite)) : NaN;
const pbcMuMedian = median(pbcMuShifts);
const pbcInteractingFinal = countPbcStatus((r) => Math.abs(num(r.U)) > 1e-12 && ['strict_final', 'sign_limited'].includes(r.status));

const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });
const C = {
  bg: '#FFFFFF', ink: '#080A0D', muted: '#5E6570', panel: '#EDEDED',
  rule: '#B8BCC4', pbc: '#6B7078', obc: '#3D8DFF', accent: '#6DCBF4',
  warm: '#D65B32', green: '#2B8A66', paleBlue: '#EAF5FC', paleWarm: '#FFF1EA',
};
const FONT = 'Helvetica Neue';

function addText(slide, text, position, style = {}) {
  const box = slide.shapes.add({
    geometry: 'textbox', position, fill: 'none',
    line: { style: 'solid', fill: 'none', width: 0 },
  });
  box.text = String(text);
  box.text.style = {
    fontFamily: FONT, fontSize: 18, color: C.ink,
    autoFit: 'shrinkText', verticalAlignment: 'middle',
    ...style,
  };
  return box;
}
function addRect(slide, position, fill = C.panel, lineFill = C.rule, lineWidth = 0) {
  return slide.shapes.add({
    geometry: 'rect', position, fill,
    line: { style: 'solid', fill: lineFill, width: lineWidth },
  });
}
function addRule(slide, left, top, width, fill = C.rule, height = 1) {
  return addRect(slide, { left, top, width, height }, fill, fill, 0);
}
function addPage(slide, page) {
  addText(slide, `${page}/18`, { left: 1166, top: 690, width: 70, height: 16 },
    { fontSize: 11, color: C.muted, alignment: 'right' });
}
function addSource(slide, text) {
  addText(slide, text, { left: 43, top: 690, width: 1090, height: 16 },
    { fontSize: 9.5, color: C.muted });
}
function addHeader(slide, title, subtitle, page) {
  slide.background.fill = C.bg;
  addText(slide, title, { left: 43, top: 18, width: 1110, height: 43 },
    { fontSize: title.length > 70 ? 34 : 37, bold: true, verticalAlignment: 'top' });
  addText(slide, subtitle, { left: 44, top: 61, width: 1175, height: 26 },
    { fontSize: 16, color: C.muted, verticalAlignment: 'top' });
  addRule(slide, 43, 87, 1194, C.rule, 1);
  addPage(slide, page);
}
function statLabel(slide, value, label, x, y, width, color = C.ink) {
  addText(slide, value, { left: x, top: y, width, height: 56 },
    { fontSize: 42, bold: true, color, verticalAlignment: 'bottom' });
  addText(slide, label, { left: x, top: y + 58, width, height: 48 },
    { fontSize: 17, color: C.muted, verticalAlignment: 'top' });
}

// 1 — title, method, and provenance.
{
  const slide = presentation.slides.add();
  slide.background.fill = C.bg;
  addText(slide, 'L=6  •  OPEN BOUNDARY CONDITIONS', { left: 45, top: 39, width: 650, height: 28 },
    { fontSize: 16, bold: true, color: C.obc, characterSpacing: 1.2 });
  addText(slide, 'CE/GCE equal-time\nthermometry without wrap bonds',
    { left: 45, top: 118, width: 720, height: 190 },
    { fontSize: 58, bold: true, verticalAlignment: 'top' });
  addText(slide,
    'A 192-condition mirror of the L=6 periodic calculation, using the validated SmoQyDQMC v2.0.12 OBC fork and density-tuned GCE production.',
    { left: 48, top: 334, width: 675, height: 94 },
    { fontSize: 23, color: C.muted, verticalAlignment: 'top' });
  addRect(slide, { left: 804, top: 96, width: 390, height: 455 }, C.panel, C.panel, 0);
  addText(slide, '36', { left: 850, top: 129, width: 280, height: 82 }, { fontSize: 74, bold: true, color: C.obc, alignment: 'center' });
  addText(slide, 'sites', { left: 850, top: 205, width: 280, height: 30 }, { fontSize: 19, color: C.muted, alignment: 'center' });
  addRule(slide, 850, 255, 280, '#C8CDD4', 1);
  addText(slide, '60  /  50', { left: 835, top: 278, width: 320, height: 70 }, { fontSize: 48, bold: true, alignment: 'center' });
  addText(slide, 'undirected NN / NNN bonds', { left: 835, top: 345, width: 320, height: 32 }, { fontSize: 18, color: C.muted, alignment: 'center' });
  addText(slide, 'T = 1/β\nΔT/TCE = (TGCE−TCE)/TCE', { left: 835, top: 416, width: 320, height: 82 },
    { fontSize: 22, bold: true, alignment: 'center' });
  addText(slide, 'CE/GCE observables use identical physical bond lists; connected charge is formed only after global phase and site-density pooling.',
    { left: 47, top: 535, width: 1130, height: 80 }, { fontSize: 19, color: C.ink, verticalAlignment: 'top' });
  addSource(slide, `Project ${S.project_commit || 'recorded in snapshot'}  •  SmoQyDQMC ${S.smoqydqmc_version || '2.0.12'} @ ${(S.smoqydqmc_commit || '').slice(0, 12)}  •  OBC normalization: 36 sites, 60 NN, 50 NNN.`);
  addPage(slide, 1);
}

function subtitleFor(row) {
  if (row.kind === 'thermometer') {
    return '(T_GCE−T_CE)/T_CE; join actual unique-root points only; break at no-T, ambiguous, flat, sign-limited, or unfinished conditions.';
  }
  if (row.kind === 'thermometry_scorecard') {
    return 'Unique-temperature coverage is shown together with no-solution, ambiguous/flat, sign-limited, and median-bias statistics.';
  }
  if (row.kind === 'mismatch') {
    return 'Matched OBC CE/GCE conditions use the actual simulation temperature T=1/β; missing or sign-limited conditions are not imputed.';
  }
  return 'Open 6×6 lattice. CE and GCE use identical site, NN, and NNN bond lists; error bars are pooled rank uncertainties.';
}
function displayTitle(row) {
  if (row.kind !== 'thermometer') return row.title;
  const compact = {
    kinetic: 'Kinetic-energy thermometry under OBC',
    double_occupancy: 'Double-occupancy thermometry under OBC',
    nn_spin: 'NN-spin thermometry under OBC',
    nn_charge_connected: 'NN connected-charge thermometry under OBC',
  };
  return compact[row.quantity] || row.title;
}

// 2–15 — the 14 analysis panels, with full-width/tall thermometers.
for (let i = 0; i < assets.length; i++) {
  const row = assets[i];
  const page = i + 2;
  const slide = presentation.slides.add();
  addHeader(slide, displayTitle(row), subtitleFor(row), page);
  const png = await fs.readFile(row.png);
  const isThermometer = row.kind === 'thermometer';
  const imageTop = isThermometer ? 91 : 94;
  const imageHeight = isThermometer ? 563 : 585;
  slide.images.add({
    blob: png.buffer.slice(png.byteOffset, png.byteOffset + png.byteLength),
    contentType: 'image/png', alt: row.title, fit: 'contain',
    position: { left: 12, top: imageTop, width: 1256, height: imageHeight },
  });
  if (isThermometer) {
    const stat = OBC.get(row.quantity);
    addRect(slide, { left: 54, top: 655, width: 1170, height: 31 }, C.paleBlue, C.paleBlue, 0);
    addText(slide,
      `Unique T ${pct(stat.unique_fraction)}   •   No inferred T ${pct(stat.no_solution_fraction)}   •   Ambiguous/flat ${pct(stat.ambiguous_or_flat_fraction)}   •   Sign-limited ${pct(stat.sign_limited_fraction)}   •   Median |ΔT|/TCE ${pct(stat.median_abs_deltaT_over_TCE)}`,
      { left: 67, top: 658, width: 1144, height: 24 },
      { fontSize: 15.5, bold: true, alignment: 'center' });
  }
  addSource(slide, 'Source: strict-final L=6 OBC CE/GCE production and exact U=0 grid; GCE rows require |N−Ntarget|≤0.03.');
}

// 16 — paired PBC/OBC thermometry comparison.
{
  const slide = presentation.slides.add();
  const coverageDelta = QORDER.map((q) => num(OBC.get(q).unique_fraction) - num(PBC.get(q).unique_fraction));
  const largest = QORDER.reduce((best, q, i) => Math.abs(coverageDelta[i]) > Math.abs(coverageDelta[best]) ? i : best, 0);
  const qLargest = QORDER[largest];
  const direction = coverageDelta[largest] >= 0 ? 'higher' : 'lower';
  addHeader(slide, 'Boundary choice changes thermometer coverage and bias',
    'Each bar uses the same L=6 physics grid and the same unique-root inversion rule; only the spatial boundary changes.', 16);
  addText(slide, 'Unique inferred-temperature coverage', { left: 55, top: 105, width: 545, height: 30 }, { fontSize: 23, bold: true });
  addText(slide, 'Median |(TGCE−TCE)/TCE| among unique roots', { left: 665, top: 105, width: 545, height: 30 }, { fontSize: 23, bold: true });
  const categories = QORDER.map((q) => QSHORT[q]);
  slide.charts.add('bar', {
    position: { left: 42, top: 140, width: 568, height: 385 }, categories,
    series: [
      { name: 'PBC', categories, values: QORDER.map((q) => Math.round(1000 * num(PBC.get(q).unique_fraction, 0)) / 10), fill: C.pbc },
      { name: 'OBC', categories, values: QORDER.map((q) => Math.round(1000 * num(OBC.get(q).unique_fraction, 0)) / 10), fill: C.obc },
    ],
    hasLegend: true, legend: { position: 'bottom', overlay: false }, dataLabels: { showValue: false },
    chartFill: C.bg, chartLine: { style: 'solid', width: 0, fill: C.bg }, plotAreaFill: { type: 'none' },
    plotAreaLine: { style: 'solid', width: 0, fill: C.bg },
    xAxis: { visible: true, line: { style: 'solid', width: 1, fill: C.rule }, textStyle: { typeface: FONT, fontSize: '13px', color: C.ink } },
    yAxis: { visible: true, min: 0, max: 100, majorUnit: 20, majorGridlines: { style: 'solid', width: 1, fill: C.panel }, line: { style: 'solid', width: 0, fill: C.bg }, textStyle: { typeface: FONT, fontSize: '13px', color: C.ink } },
    barOptions: { direction: 'column', grouping: 'clustered', gapWidth: 80 },
  });
  const allBias = [...QORDER.map((q) => 100 * num(PBC.get(q).median_abs_deltaT_over_TCE, 0)), ...QORDER.map((q) => 100 * num(OBC.get(q).median_abs_deltaT_over_TCE, 0))];
  const biasMax = Math.max(5, Math.ceil(Math.max(...allBias) / 5) * 5);
  slide.charts.add('bar', {
    position: { left: 652, top: 140, width: 586, height: 385 }, categories,
    series: [
      { name: 'PBC', categories, values: QORDER.map((q) => Math.round(1000 * num(PBC.get(q).median_abs_deltaT_over_TCE, 0)) / 10), fill: C.pbc },
      { name: 'OBC', categories, values: QORDER.map((q) => Math.round(1000 * num(OBC.get(q).median_abs_deltaT_over_TCE, 0)) / 10), fill: C.obc },
    ],
    hasLegend: true, legend: { position: 'bottom', overlay: false }, dataLabels: { showValue: false },
    chartFill: C.bg, chartLine: { style: 'solid', width: 0, fill: C.bg }, plotAreaFill: { type: 'none' },
    plotAreaLine: { style: 'solid', width: 0, fill: C.bg },
    xAxis: { visible: true, line: { style: 'solid', width: 1, fill: C.rule }, textStyle: { typeface: FONT, fontSize: '13px', color: C.ink } },
    yAxis: { visible: true, min: 0, max: biasMax, majorGridlines: { style: 'solid', width: 1, fill: C.panel }, line: { style: 'solid', width: 0, fill: C.bg }, textStyle: { typeface: FONT, fontSize: '13px', color: C.ink } },
    barOptions: { direction: 'column', grouping: 'clustered', gapWidth: 80 },
  });
  addRect(slide, { left: 55, top: 548, width: 1168, height: 109 }, C.panel, C.panel, 0);
  addText(slide,
    `${QSHORT[qLargest]} has the largest boundary-dependent coverage shift: OBC is ${Math.abs(100 * coverageDelta[largest]).toFixed(1)} percentage points ${direction}. Compare bias only where each calibration has a unique root; unresolved points remain explicit rather than interpolated through.`,
    { left: 78, top: 565, width: 1120, height: 72 }, { fontSize: 21, bold: true, verticalAlignment: 'middle' });
  addSource(slide, 'Source: final L=6 PBC and OBC thermometry summaries; identical observable definitions and ΔT/TCE convention.');
}

// 17 — boundary-dependent sign, density, and tuning scorecard.
{
  const slide = presentation.slides.add();
  addHeader(slide, 'OBC density tuning closes every GCE target without hiding sign limits',
    'PBC supplies the central chemical-potential seed; OBC retunes independently and CE positive-U pilots are classified by their own phase.', 17);
  addText(slide, 'Positive-U β≥5 CE', { left: 58, top: 121, width: 350, height: 34 }, { fontSize: 24, bold: true });
  addText(slide, 'GCE achieved-N control', { left: 462, top: 121, width: 350, height: 34 }, { fontSize: 24, bold: true });
  addText(slide, 'Chemical-potential retuning', { left: 866, top: 121, width: 350, height: 34 }, { fontSize: 24, bold: true });
  addRule(slide, 432, 115, 1, C.rule, 430); addRule(slide, 836, 115, 1, C.rule, 430);
  statLabel(slide, `${pbcPilotAdmitted}/${pbcPilotRows.length || 24}`, 'PBC admitted pilots', 58, 170, 315, C.pbc);
  statLabel(slide, `${num(S.positive_high_beta_admitted, 0)}/${num(S.positive_high_beta_pilots, 24)}`, 'OBC admitted pilots', 58, 294, 315, C.obc);
  addText(slide, `Sign-limited: PBC ${pbcPilotLimited}  •  OBC ${num(S.positive_high_beta_sign_limited, 0)}`,
    { left: 58, top: 425, width: 330, height: 52 }, { fontSize: 18, bold: true });
  addText(slide, `Median |phase| for completed OBC positive-U β≥5 rows: ${fixed(S.positive_high_beta_phase_median_abs, 4)}`,
    { left: 58, top: 485, width: 330, height: 55 }, { fontSize: 16.5, color: C.muted, verticalAlignment: 'top' });

  statLabel(slide, fixed(pbcDensityMedian, 4), 'PBC median |N−Ntarget|', 462, 170, 315, C.pbc);
  statLabel(slide, fixed(S.gce_density_median_abs_error, 4), 'OBC median |N−Ntarget|', 462, 294, 315, C.obc);
  addText(slide, `Worst case: PBC ${fixed(pbcDensityMax, 4)}  •  OBC ${fixed(S.gce_density_max_abs_error, 4)}  •  tolerance 0.0300`,
    { left: 462, top: 425, width: 330, height: 58 }, { fontSize: 18, bold: true });
  addText(slide, `${num(S.gce_density_within_0p03, 0)}/${num(S.interacting_gce_rows, 152)} interacting OBC GCE targets meet the production tolerance.`,
    { left: 462, top: 490, width: 330, height: 55 }, { fontSize: 16.5, color: C.muted, verticalAlignment: 'top' });

  statLabel(slide, fixed(pbcMuMedian, 3), 'PBC median |μL6−μL8 seed|', 866, 170, 315, C.pbc);
  statLabel(slide, fixed(S.gce_mu_shift_from_pbc_median_abs, 3), 'OBC median |μOBC−μPBC seed|', 866, 294, 315, C.obc);
  addText(slide, `${num(S.gce_pbc_seed_provenance_rows, 0)}/${num(S.interacting_gce_rows, 152)} OBC rows retain their PBC seed and inherited L=8 provenance.`,
    { left: 866, top: 425, width: 330, height: 58 }, { fontSize: 18, bold: true });
  addText(slide, `OBC geometry: ${S.site_count} sites, ${S.nn_bond_count} NN bonds, ${S.nnn_bond_count} NNN bonds; no periodic estimators.`,
    { left: 866, top: 490, width: 330, height: 55 }, { fontSize: 16.5, color: C.muted, verticalAlignment: 'top' });
  addRect(slide, { left: 58, top: 575, width: 1138, height: 72 }, C.paleBlue, C.paleBlue, 0);
  addText(slide,
    `Terminal coverage — PBC interacting ensemble rows: ${pbcInteractingFinal}/304. OBC: ${num(S.interacting_ce_strict_final, 0) + num(S.interacting_ce_sign_limited, 0) + num(S.interacting_gce_strict_final, 0)}/304. Exact U=0: ${S.exact_u0_ensemble_rows}/80 ensemble rows.`,
    { left: 78, top: 592, width: 1098, height: 38 }, { fontSize: 20, bold: true, alignment: 'center' });
  addSource(slide, 'Source: final L=6 PBC/OBC condition-status and production snapshots; sign-limited CE rows remain terminal but are never assigned thermometry values.');
}

// 18 — synthesis and guardrails.
{
  const slide = presentation.slides.add();
  const ranked = [...QORDER].sort((a, b) => {
    const du = num(OBC.get(b).unique_fraction) - num(OBC.get(a).unique_fraction);
    if (Math.abs(du) > 1e-12) return du;
    return num(OBC.get(a).median_abs_deltaT_over_TCE, Infinity) - num(OBC.get(b).median_abs_deltaT_over_TCE, Infinity);
  });
  const best = ranked[0];
  slide.background.fill = C.bg;
  addText(slide, 'CONCLUSION', { left: 45, top: 40, width: 250, height: 28 },
    { fontSize: 16, bold: true, color: C.obc, characterSpacing: 1.2 });
  addText(slide, `${QSHORT[best]} gives the strongest L=6 OBC thermometry coverage`,
    { left: 45, top: 104, width: 1135, height: 105 }, { fontSize: 49, bold: true, verticalAlignment: 'top' });
  addText(slide,
    `Its OBC calibration yields a unique inferred temperature for ${pct(OBC.get(best).unique_fraction)} of actual CE conditions, with median |ΔT|/TCE ${pct(OBC.get(best).median_abs_deltaT_over_TCE)} among those unique roots.`,
    { left: 48, top: 230, width: 1120, height: 82 }, { fontSize: 24, color: C.muted, verticalAlignment: 'top' });
  addRule(slide, 47, 342, 1160, C.rule, 1);
  const points = [
    ['Boundary is part of the calibration', 'PBC and OBC use the same grid and observable definitions, yet their unique-root coverage and inferred bias need not agree.'],
    ['Density matching is explicit', `${num(S.gce_density_within_0p03, 0)}/${num(S.interacting_gce_rows, 152)} interacting OBC GCE rows satisfy |N−Ntarget|≤0.03 after PBC-seeded retuning.`],
    ['Unresolved temperatures stay unresolved', 'No-solution, multiple-root, flat-calibration, unfinished, and sign-limited points break the line and are excluded from bias statistics.'],
  ];
  for (let i = 0; i < points.length; i++) {
    const x = 48 + i * 400;
    addText(slide, points[i][0], { left: x, top: 379, width: 350, height: 55 }, { fontSize: 23, bold: true, verticalAlignment: 'top' });
    addText(slide, points[i][1], { left: x, top: 446, width: 350, height: 126 }, { fontSize: 18, color: C.muted, verticalAlignment: 'top' });
  }
  addRect(slide, { left: 48, top: 604, width: 1156, height: 59 }, C.panel, C.panel, 0);
  addText(slide, 'Interpretation rule: thermometry claims are observable- and boundary-specific; use the coverage scorecard together with the reported ΔT/TCE distribution.',
    { left: 70, top: 617, width: 1112, height: 32 }, { fontSize: 20, bold: true, alignment: 'center' });
  addSource(slide, 'Source: final L=6 PBC/OBC equal-time thermometry outputs and strict validation ledgers.');
  addPage(slide, 18);
}

if (presentation.slides.items.length !== 18) throw new Error(`expected 18 slides, found ${presentation.slides.items.length}`);
const inspection = await presentation.inspect({ kind: 'slide,textbox,shape,image,chart,layout', maxChars: 500000 });
await fs.writeFile(`${outputPath}.inspect.ndjson`, inspection.ndjson);
const output = await PresentationFile.exportPptx(presentation);
await output.save(outputPath);
console.log(outputPath);
