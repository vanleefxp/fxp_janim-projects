

+ $0 in NN$
+ $forall m in NN " " (m = m)$
+ $forall m in NN " " forall n in NN " " (m = n -> n = m)$
+ $forall m in NN " " forall n in NN " " forall t in NN " " ((m = n and n = t) -> m = t)$
+ $forall m " " forall n " " ((n in NN and m = n) -> m in NN)$
+ $forall m in NN " " (S(m) in NN)$
+ $forall m in NN " " forall n in NN " " (S(m) = S(n) -> m = n)$
+ $forall m in NN " " (S(n) != 0)$
+ $forall K " " ((0 in K and (forall n in NN " " (n in K -> S(n) in K))) -> K = NN)$