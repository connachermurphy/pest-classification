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

#let name = "baseline_220407"
= Summary: #name
#include("validation_results/" + name + "/summary.typ")
