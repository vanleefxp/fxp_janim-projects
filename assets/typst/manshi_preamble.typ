#import "@janim/colors:0.0.0": *

#set text(
  font: (
    "New Computer Modern",
    "FandolSong",
  ),
  lang: "zh",
)
// #show strong: set text(
//   font: (
//     "New Computer Modern",
//     "FandolHei"
//   )
// )
#set par(leading: 1em, spacing: 1.5em, justify: true)
#set table(
  stroke: 0.5pt + white,
  align: center + horizon,
)
#show sym.gt.eq: sym.gt.eq.slant
#show sym.lt.eq: sym.lt.eq.slant
#show terms: it => {
  grid(
    ..it.children
      .map(item => (strong(item.term), item.description))
      .join(),
    columns: (auto, 1fr),
    row-gutter: if (it.tight) { par.leading } else { par.spacing },
    column-gutter: 1em,
    align: top + start,
  )
}

#let boldup(body) = math.bold(math.upright(body))