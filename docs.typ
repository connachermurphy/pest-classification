#import("typst/template_paper.typ"): *
#import "@preview/tablex:0.0.5": tablex, cellx, hlinex, vlinex

// #set math.equation(numbering: "(1)")
// #set figure.caption(position: top)

// #show raw.where(block: true): block.with(
//   fill: luma(250),
//   stroke: luma(100) + 1pt,
//   inset: 6pt,
//   radius: 4pt,
// )

// #show quote: set align(center)

#show: paper.with(
  title: [Crop Pest and Pathogen Detection with Computer Vision],
  authors: (
    "Connacher Murphy",
  ),
  date: datetime.today().display("[month repr:long] [day], [year]"),
)

= Introduction
We develop a crop pest and pathogen diagnostic model. We train a neural network on the Mensah et al. (2023) CCMT dataset.

Our diagnostic model powers the `CroPP` tool at #link("https://saxifrage.co")[saxifrage].

= Data
We use the Mensah et al. (2023) CCMT dataset. We assign the images into training and test sets with probabilities 80% and 20%, respectively.

The data include labelled images of corn, cassava, maize, and tomatoes. We use only the maize data at present.

// == Maize (create a table with class frequencies)

= Architecture
We begin with a pre-trained instance of `resnet18`. We then conduct further training for each crop. We adopt a cross entropy loss function.

= Cross-Validation Exercises
We consider the selection of hyperparameters through 5-fold cross-validation.

= Deployment Training
The live version of `CroPP` is trained with
- $30$ epochs,
- a $1 times 10^(-4)$ learning rate,
- and a batch size of $128$.

You can view a demonstration of the detection model on the #link("https://saxifrage.co")[saxifrage website].

// The leaf spot mislabels:
// /Users/connormurphy/data/ccmt/Raw Data/CCMT Dataset/Maize/fall armyworm/fall armyworm835_.jpg /Users/connormurphy/data/ccmt/Raw Data/CCMT Dataset/Maize/fall armyworm/fall armyworm802_.jpg /Users/connormurphy/data/ccmt/Raw Data/CCMT Dataset/Maize/fall armyworm/fall armyworm803_.jpg /Users/connormurphy/data/ccmt/Raw Data/CCMT Dataset/Maize/fall armyworm/fall armyworm804_.jpg /Users/connormurphy/data/ccmt/Raw Data/CCMT Dataset/Maize/fall armyworm/fall armyworm805_.jpg /Users/connormurphy/data/ccmt/Raw Data/CCMT Dataset/Maize/fall armyworm/fall armyworm806_.jpg /Users/connormurphy/data/ccmt/Raw Data/CCMT Dataset/Maize/fall armyworm/fall armyworm832_.jpg /Users/connormurphy/data/ccmt/Raw Data/CCMT Dataset/Maize/fall armyworm/fall armyworm833_.jpg /Users/connormurphy/data/ccmt/Raw Data/CCMT Dataset/Maize/fall armyworm/fall armyworm834_.jpg