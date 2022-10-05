
## Problem II

Notation: $u\sim\mathcal{N}(c,I_q), c\in R^q,A\in R^{d\times q}$

### 1)

First, calculate expectation of $A_{ij}$
\begin{align*}
E[A_{ij}]&=1\times p+0\times(1-2p)+(-1)\times p\\
&=0
\end{align*}

Then we have $E[A]=\left[E[A_{ij}]\right]_{d\times q}=0_{d\times q}$ as entries $A_{ij}$ in $A$ are independent of each other. In addtion, $A$ and $u$ are independent so we can switch the order between expecation and multiplication, that is,
\begin{align*}
E[Au]&=E[A]E[u]\\
&=0_{d\times q}c_{q\times1}\\
&=0_{d\times 1}
\end{align*}
Similarly,
\begin{align*}
E[A(u-c)]&=E[Au]-E[Ac]\\
&=E[A]E[u]-E[A]c\\
&=E[A]c-E[A]c\\
&=0_{d\times 1}
\end{align*}

### 2)
Note that $A$ and $A^T$ are not indepent as at least they share same diagonal entries. Let $C=AA^T$. Then
\begin{align*}
C_{ii}&=\sum_{j=1}^dA_{ij}(A^T)_{ji}\\
&=\sum_{j=1}^dA_{ij}^2
\end{align*}
From the probability mass function (pmf) of $A_{ij}$, we derive the pmf of $A_{ij}^2$,
$$A_{ij}^2=\begin{cases}
1&\text{with probability} 2p\\
0&\text{with probability }1-2p
\end{cases}$$
Then
\begin{align*}
E[A_{ij}^2]&=1\times(2p)+0\times(1-2p)\\
&=2p
\end{align*}
Since entries $A_{ij}$ are independent, $A_{ij}^2$ are independent, too.
\begin{align*}
E[C_{ii}]&=E\left[\sum_{j=1}^dA_{ij}^2\right]\\
&=\sum_{j=1}^dE\left[A_{ij}^2\right]\\
&=\sum_{j=1}^d2p\\
&=2dp
\end{align*}
For any $i,j$ with $\neq j$
\begin{align*}
C_{ij}&=\sum_{k=1}^dA_{ik}(A^T){kj}\\
&=\sum_{k=1}^dA_{ik}A_{jk}
\end{align*}
and then
\begin{align*}
E[C_{ij}]&=E\left[\sum_{k=1}^dA_{ik}A_{jk}\right]\\
&=\sum_{k=1}^dE[A_{ik}A_{jk}]\\
&\quad A_{ik},A_{jk}\text{ are independent}\\
&=\sum_{k=1}^dE[A_{ik}]E[A_{jk}]\\
&=\sum_{k=1}^d0\cdot0\\
&=0
\end{align*}
Therefore,
$$E[AA^T]=2dp I_{d}$$
where $I_d$ is the identity matrix of size $d\times d$.



