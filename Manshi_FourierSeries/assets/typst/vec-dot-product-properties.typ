
#let vecu = $text(bold(upright(u)), fill: #rgb("#FC6255"))$
#let vecv = $text(bold(upright(v)), fill: #rgb("#83C167"))$


#context grid(
  [双线性性],
  $
    angle.l k#vecu, #vecv angle.r
    &= angle.l #vecu, k #vecv angle.r \
    &= k angle.l #vecv, #vecu angle.r
  $,
  [对称性],
  $
    angle.l #vecu, #vecv angle.r
    = angle.l #vecv, #vecu angle.r
  $,
  [正定性],
  $
    &angle.l #vecu, #vecu angle.r >= 0 \
    &angle.l #vecu, #vecu angle.r = 0 <=> #vecu = bold(0)
  $,
  row-gutter: (par.leading, 2em) * 3,
)
