### E-propにおける$d$と$\partial$の使い分け
- $\frac{d E}{d z_j^t}$は，接続されているすべてのニューロンの，時刻$t+1$以降のスパイク$z_j$への影響を考慮した全微分
- $\frac{\partial E}{\partial z_j^t}$は，$z_j^t$自身と，$z_j^t$自身が直接与える変数への影響のみを考慮する偏微分

### 学習シグナル$L_t$について
- エネルギー関数$E$はすべての時刻の和をとるため，ある時刻$t$のスパイク$z_j^t$で偏微分する場合，$z_j^t$が未来のステップに及ぼす影響を考慮するため，次のように連鎖律が決まる．
$$\frac{\partial E}{\partial z_j^t}=\sum_{t'=t}^{T}\sum_k\frac{\partial E}{\partial y_k^{t'}}\frac{\partial y_k^{t'}}{\partial z_j^t}.$$
- したがって二重フィルタを含むPredictive E-propの場合は次のようになる．
$$\frac{\partial E}{\partial z_j^t}=\sum_{t'=t}^{T}\frac{\partial E}{\partial\bar{\bar{z}}_j^{t'}}\frac{\partial\bar{\bar{z}}_j^{t'}}{\partial z_j^t}.$$