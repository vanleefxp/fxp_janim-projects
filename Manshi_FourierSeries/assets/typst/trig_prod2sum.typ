#import "@janim/colors:0.0.0": *

#{
  show sym.alpha: text.with(fill: RED)
  show sym.beta: text.with(fill: GREEN)
  $
    &cos(alpha) &&sin(beta) &&= &&1/2 (&&sin(alpha + beta) &&- &&sin(alpha - beta)) \
    &cos(alpha) &&cos(beta) &&= &&1/2 (&&cos(alpha + beta) &&+ &&cos(alpha - beta)) \
    &sin(alpha) &&sin(beta) &&= &-&1/2 (&&cos(alpha + beta) &&- &&cos(alpha - beta))
  $
}