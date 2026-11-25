### E-propにおける$d$と$\partial$の使い分け
- $\frac{d E}{d z_j^t}$は，接続されているすべてのニューロンの，時刻$t+1$以降のスパイク$z_j$への影響を考慮した全微分
- $\frac{\partial E}{\partial z_j^t}$は，$z_j^t$自身と，$z_j^t$自身が直接与える変数への影響のみを考慮する偏微分

### 学習シグナル$L_t$について
- エネルギー関数$E$はすべての時刻の和をとるため，ある時刻$t$のスパイク$z_j^t$で偏微分する場合，$z_j^t$が未来のステップに及ぼす影響を考慮するため，次のように連鎖律が決まる．
$$\frac{\partial E}{\partial z_j^t}=\sum_{t'=t}^{T}\sum_k\frac{\partial E}{\partial y_k^{t'}}\frac{\partial y_k^{t'}}{\partial z_j^t}.$$
- したがって二重フィルタを含むPredictive E-propの場合は次のようになる．
$$\frac{\partial E}{\partial z_j^t}=\sum_{t'=t}^{T}\frac{\partial E}{\partial\bar{\bar{z}}_j^{t'}}\frac{\partial\bar{\bar{z}}_j^{t'}}{\partial z_j^t}.$$

### 連鎖律
- ある状態に対するある状態の全微分を考えるときは，微分する側の状態から直接行ける状態の数だけ分解をする
- 直接行ける状態なので，連鎖律で分解したときに初めの項に全微分を押し付ければ，第2項は直接のみ考えればよく，偏微分となる