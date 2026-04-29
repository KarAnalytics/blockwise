## R CMD check results

0 errors | 0 warnings | 1 note

The single NOTE is from `* checking CRAN incoming feasibility` and contains
the two items expected for a first submission:

* `New submission` — this is the package's first release on CRAN.
* `Possibly misspelled words in DESCRIPTION` — all flagged words are
  intentional and correctly spelled proper nouns:
    * `BRM` — acronym for the method (Blockwise Reduced Modeling).
    * `Blockwise`, `blockwise` — the method and package name.
    * `Currim`, `Srinivasan` — author surnames.

## Test environments

* win-builder, R-release: R 4.6.0 (2026-04-24) — 1 NOTE (as above).
* win-builder, R-devel: R Under development (unstable, r89972, 2026-04-28)
  — 1 NOTE (as above).

## Downstream dependencies

There are currently no downstream dependencies (this is a first release).
