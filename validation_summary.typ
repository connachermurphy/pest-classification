#import("typst/template_paper.typ"): *

#set enum(indent: 5pt)
#set list(indent: 5pt)

#show: paper.with(
  title: [Validation Results],
  authors: (
    "Connacher Murphy",
  ),
  date: datetime.today().display("[month repr:long] [day], [year]"),
)

This document contains results from cross-validation exercises.

= Summary: `baseline_220328`
#include("validation_results/baseline_220328/summary.typ")
