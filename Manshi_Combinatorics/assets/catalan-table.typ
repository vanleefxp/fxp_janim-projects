#let n = 10
#let indices = range(n + 1)
#let values = ()
#for i in indices {
  if i == 0 {
    values = (1,)
  } else {
    let prev = values.at(i - 1)
    let current = calc.div-euclid(prev * (2 * (2 * i - 1)), (i + 1))
    values = (..values, current)
  }
}

#table(
  columns: /*(1fr,) * */(n + 2),
  stroke: 1pt + white,
  align: center,
  [$n$],
  ..indices.map(i => [$#i$]),
  [$a_n$],
  ..indices.map(i => [$#values.at(i)$])
)
