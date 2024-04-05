#import("typst/template_paper.typ"): *

#show: paper.with(
  title: [Validation Results],
  authors: (
    "Connacher Murphy",
  ),
  date: datetime.today().display("[month repr:long] [day], [year]"),
)

= `baseline_220328`
#include("validation_results/baseline_220328/summary.typ")
