# Qualifying classification and usable pace

The qualifying source adapter separates classification status, classification
rank, observed lap times, and whether those times may supply pace. Some Jolpica
qualifying records give disqualified drivers ordinary numeric positions, including
duplicates, while retaining laps that must not be treated as valid qualifying
performance.

| Primary qualifying evidence | Effective rank | Effective stage times |
| --- | --- | --- |
| Numeric classified position | Published position | Explicit valid stage times |
| DQ/DSQ | Missing | Missing; displayed/provider times remain in evidence |
| NC/DNQ | Missing | Explicit observed stage times may remain usable |
| DNS/DNP or an explicit same-driver no-participation note | Missing | Missing |
| Missing or ambiguous evidence | Unknown | No invented classification or laps |

An untimed entrant is not automatically disqualified. A classified driver with a
numeric position can have no stage times; a driver marked NC can have valid laps.
The official race grid and pit queue remain separate from all these fields.

## Live recovery and saved evidence

The existing current-season official qualifying fallback now also handles
ambiguous provider ranks. It resolves the event from the official results index
and reads one GP qualifying table. Driver codes, supplied car numbers,
constructors, season, event, and GP session must match. Known circuit slugs handle
names such as Dutch/Netherlands and British/Great Britain. Constructor validation
recognizes chassis names before engine suffixes such as Haas Ferrari and Alfa
Romeo Ferrari.

The adapter preserves original provider rows and adds optional versioned
`qualifying_classification` metadata. This retains reported fields, official
display fields, effective rank and per-stage validity, driver/constructor identity,
and the primary URL and content digest. A primary partial classification must
cover the complete known field from one event, URL, and digest. A missing table
row needs an explicit matching official note; an absent provider identity is not
created from the grid. Unresolved ambiguous classifications fail closed.

Confirmed-grid snapshots retain the reported qualifying fields in their source
context and derive the effective feature values during rebuilding. Saved replay
checks reported fields against the retained row, effective values against the
official display, the joined field and event identity, and an available source
receipt fetched by the forecast cutoff. All required source reads precede the
default cutoff capture. Digests detect changes; they do not authenticate historical
publication times.

The same validity rules apply to target and prior qualifying inputs, native pace,
team/driver qualifying form, qualifying calibration Q1, and practice-based Q1
history. Race finishing positions and race nonfinish statuses are untouched.

## Verified historical examples

The bounded source audit covered every older duplicate-rank event in the retained
dataset, using these four primary tables:

- [2023 Great Britain qualifying](https://www.formula1.com/en/results/2023/races/1214/great-britain/qualifying): Bottas DQ.
- [2024 Monaco qualifying](https://www.formula1.com/en/results/2024/races/1236/monaco/qualifying): Hulkenberg and Magnussen DQ.
- [2024 Netherlands qualifying](https://www.formula1.com/en/results/2024/races/1243/netherlands/qualifying): Albon DQ; Sargeant explicitly did not take part.
- [2024 Azerbaijan qualifying](https://www.formula1.com/en/results/2024/races/1245/azerbaijan/qualifying): Gasly DQ.

The source-only replay retained all 224 eligible events. Thirteen feature matrices
changed through target qualifying or strictly earlier same-season form. All risk
matrices, the original 105 training matrices, and the current 40 target matrices
were exact. Removing the metadata recovered the original data exactly.

## Scope and accuracy

Legacy saved sources without the new metadata preserve their original numerical
interpretation, including schema-5 offline forecast replay. The feature count,
model format, fitted artifacts, and statistical specification are unchanged by
this source correction.

The live ambiguity trigger does not certify every provider-only classification:
a hidden DQ can retain a unique numeric rank. The audit also found genuinely absent
or untimed older qualifying observations; it does not claim that all such cases
can be reconstructed. Historical sources fetched after races establish a logical
information cutoff, not proof of their original publication state.

The separate fixed-specification check retrained before each of 63 historical
races, using only strictly earlier completed events, and saved all forecasts
before scoring. The corrected older inputs improved raw analytic winner Brier
in 2024 from 0.73461 to 0.73091. On the current 40 races it changed 0.51036 to
0.51034, with the same 28 correct picks, slightly worse log loss, and a small
2026 regression. The candidate failed its accuracy screen; no trained serving
artifact was replaced. These analytic scores precede native-path weighting and
are distinct from the shipped experimental forecast's historical scores.

The [compact audit receipt](../evidence/qualifying-classification-2026-10-18.json)
retains source corrections, original-matrix parity and the assessment. Full
local source pages, feature matrices, plans, models and independent audits remain
under `output/qualifying-classification-history-2026-10-18/`. This is a verified
input and replay correction; it does not establish an additional accuracy gain.
