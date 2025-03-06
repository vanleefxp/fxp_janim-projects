$
  1 - sqrt(1 - 4x) 
  &= - &&sum_(n=1)^(infinity) binom("1/2", n)(-4x)^n quad
  &&= - &&sum_(n=1)^(infinity) ((-1)^(n - 1) (2n - 3)!!)/(2^n n!)(-4x)^n \ 
  &= &&sum_(n=1)^(infinity) ((-1)^n (2n-1)!!)/(2^(n + 1) (n + 1)!)(-4x)^(n + 1) quad
  &&= &&sum_(n=0)^(infinity) (2^(n + 1) (2n - 1)!!)/((n + 1)!) x^(n + 1) \ 
  &= &&sum_(n = 0)^(infinity) (2(2n)!)/((n + 1)!n!) x^(n + 1) quad
  &&= &&sum_(n = 0)^(infinity) 2/(n + 1) binom(2n, n) x^(n + 1) &
$

