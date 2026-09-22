# SNN-FS and TrajSNN method definition and scope

SNN-FS and TrajSNN are training-observation weighting frameworks for sparse
feature selection. They are not neural networks. Given observations \(x_i\),
visit times \(t_i\), and outcome \(y_i\), they compute weights on the
outer-training set and fit a weighted sparse model

\[
\hat\beta=\arg\min_\beta \frac{1}{n}\sum_i w_i
\ell(y_i,x_i^\top\beta)+\lambda\{\alpha\|\beta\|_1+
(1-\alpha)\|\beta\|_2^2/2\}.
\]

For a k-nearest-neighbour set \(N_k(i)\), SNN measures local support as

\[
w_i^{SNN}=\mathrm{scale}\left[k^{-1}\sum_{j\in N_k(i)}
|N_k(i)\cap N_k(j)|\right].
\]

SNN-FS uses shared-neighbor support alone. TrajSNN multiplies this support by
the mean temporal-proximity kernel over molecular neighbours,

\[
w_i^{time}=|N_k(i)|^{-1}\sum_{j\in N_k(i)}\exp(-|t_i-t_j|/\lambda),
\qquad \lambda=0.25,
\]

after normalizing time within the training fold. Participant identity is used
only for leakage-safe splitting and is not used to define observation weights.
If a fold has no time variation, TrajSNN reduces to SNN-FS. A participant with
one visit remains eligible.

The contribution is temporal-proximity-aware neighbourhood weighting. All
selectors use the same candidate panel sizes \(\{3,5,10\}\) so that method
comparisons do not confound ranking quality with the number of retained
features.

The graph calculation is \(O(npd+n k^2)\) after transformation, where \(n\) is
the number of observations, \(p\) the original dimension, \(d\) the embedding
dimension, and \(k\) the neighbourhood size. Memory is \(O(np+n^2)\) in the
current dense implementation. The framework does not guarantee causal
biomarkers or transportability across assays.

All graphs, imputers, scalers, feature rankings, panel sizes, and prediction
penalties must be fitted in the training portion of participant-disjoint
repeated nested evaluation. External labels must not influence the locked
panel.
