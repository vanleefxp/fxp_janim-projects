#import "@janim/colors:0.0.0": *

// #let RED = red
// #let GREEN = green
// #let BLUE = blue

// #show regex(".+"): it => {
//   set text(stroke: 0.25pt + text.fill)
//   it
// }

#{
  let m = text($m$, fill: RED)
  let n = text($n$, fill: GREEN)
  let k = text($k$, fill: BLUE)
  $
    integral_0^(2 pi) sin(#k x) dif x = 0 #h(4em) integral_0^(2 pi) cos(#k x) dif x = cases(2 pi quad &"if" #k = 0, 0 quad &"otherwise")
  $

  $
    &integral_0^(2 pi) &&cos(#m x) &&cos(#n x) dif x
    &&= &&1/2 integral_0^(2 pi) (&&cos((#m + #n) x) &&+ &&cos((#m - #n) x)) dif x
    &&= cases(pi quad &"if" #m = #n, 0 quad &"otherwise") \

    &integral_0^(2 pi) &&sin(#m x) &&sin(#n x) dif x
    &&= &-&1/2 integral_0^(2 pi) (&&cos((#m + #n) x) &&- &&cos((#m - #n) x)) dif x
    &&= cases(pi quad &"if" #m = #n, 0 quad &"otherwise") \

    &integral_0^(2 pi) &&cos(#m x) &&sin(#n x) dif x
    &&= &&1/2 integral_0^(2 pi) (&&sin((#m + #n) x) &&- &&sin((#m - #n) x)) dif x
    &&= 0 \
  $
}
