# Chat Transcript: Precossモデル解説

## User

モデル "Precoss" について解説してください．

## Aurelle

# Precoss モデルの解説

**Precoss**（**Pre**dictive **co**ding and o**s**cillations for **s**peech）は、連続した自然発話から音節をリアルタイムに認識するための神経計算モデルです。その核心的なアイデアは、**予測符号化（Predictive Coding）** と **神経振動（Neural Oscillations）** という、従来は別々に研究されてきた二つのアプローチを統合することにあります。

---

## 背景：なぜこの二つを組み合わせるのか

自然発話の処理には、大きく分けて二つの側面があります。

- **ボトムアップ処理**：音響信号を逐次的にパースして音節の境界を検出する（シータ・ガンマ振動が担うとされる）
- **トップダウン処理**：文脈や内部知識から次の音節を予測する（予測符号化が担うとされる）

先行研究では、これらはそれぞれ独立したモデルとして実装されており、「連続発話のパースには強いが認識が弱い」あるいは「孤立した音声アイテムの認識には強いが連続パースができない」という限界がありました。Precossはこの両者を統合することで、**連続発話からのオンライン音節認識**を実現しています。

---

## モデルのアーキテクチャ

Precossは**二階層の生成モデル**（generative model）として構築されており、感覚入力の"隠れた原因"を推論する（＝生成モデルを反転する）ことで動作します。

### 入力

- **聴覚スペクトログラム**（6チャンネル、150 Hz〜5 kHz）：音の周波数・時間情報
- **緩やかな振幅変調**（slow amplitude modulation）：音節境界の手がかりとなる包絡線情報

### 上位レベル（Top Level）：「いつ」と「何を」の分離

モデルの最も重要な設計思想は、**"what"（何の音節か）** と **"when"（いつ・どのくらいの長さか）** を明示的に分離している点です。

#### 1. シータモジュール（Theta Module）― "When"

- **Ermentrout-Kopell の正準モデル**（canonical theta neuron）を用いてシータ振動（〜5 Hz）を実装
- 緩やかな振幅変調を入力として受け取り、特定の位相に達するたびに**ガウスパルス（シータトリガー）** を生成
- このトリガーが**音節の開始タイミング（onset）** の推定値となる
- シータ振動の瞬時周波数が、ガンマシーケンスの持続時間（＝音節の長さの推定）を動的に調整する

> "Whenever the theta oscillator reaches a predefined phase, the model generates a Gaussian pulse, referred to as theta trigger... Depending on the input, theta triggers appear sooner or later and constitute the model's estimates of syllable onsets." [Theta Module](https://alphaxiv.org/abs/2006.10800?page=3)

#### 2. スペクトル時間モジュール（Spectrotemporal Module）― "What"

- **ガンマユニット（8個）**：音節内の時間的位相を表現し、順次活性化する（ヘテロクリニックチャネルとして実装）。音節全体を8等分したスペクトルのスナップショットを順番に展開する役割を担う
- **音節ユニット（syllable units）**：各音節のスペクトル時間パターンを内部表現として保持するユニット群。文中の音節数分だけ用意される

ガンマユニットが順次活性化することで、「この音節はこういうスペクトル時間パターンを持つはず」というトップダウン予測が生成され、実際の入力スペクトログラムと比較されます。

### 下位レベル（Bottom Level）：感覚予測の生成

- **Hopfieldアトラクタネットワーク**を使って、上位レベルから渡されたカコーサル状態（音節・ガンマユニットの情報）をもとに、各周波数チャンネルの**予測スペクトログラム**を生成
- この予測と実際の感覚入力との差分が**予測誤差（prediction error）** として上位レベルにフィードバックされ、推論が更新される

---

## シータ・ガンマ結合（Theta-Gamma Coupling）の役割

シータとガンマの結合は、モデルの時間的整合性を担う中核機構です。

- **シータ→ガンマへのリセット**：シータトリガーが発生するたびに、ガンマシーケンスが最初（第1ユニット）からリセットされ、音節の開始に合わせて予測が再展開される
- **シータによるガンマの速度制御**：シータの瞬時周波数（$s_\theta$）がガンマシーケンスの速度を制御し、長い音節には長いガンマシーケンス、短い音節には短いシーケンスを自動調整する

$$\kappa_2(s) = \kappa_0 e^{(s-1)}, \quad f(s) = s_\theta - s \text{（外因性結合の場合）}$$

---

## モデルの変形バリアント（A〜F）と性能比較

[Model Variants](https://alphaxiv.org/abs/2006.10800?page=4)

Precossは合計6つのバリアントで評価されました。主な制御変数は以下の3点です：

| 変数 | 内容 |
|---|---|
| ガンマリセット | シータトリガーによるリセットあり vs なし |
| 音節ユニットリセット | 第8ガンマユニットによる証拠累積リセットあり vs なし |
| ガンマ速度制御 | 外因性（シータ駆動）/ 内因性（自律的）/ なし |

性能比較の結果をまとめると：

<!-- Bar Chart: Precoss モデルバリアント別 音節認識率（中央値） -->

> "Relative to models A and B, performance was significantly lower in models E and F (on average by 23%), and C and D (by 15%) indicating that erasing accumulated evidence about the previous syllable before processing a new syllable is a crucial factor." [Performance Comparison](https://alphaxiv.org/abs/2006.10800?page=5)

### 重要な発見

1. **最も重要な要因：音節ユニットのリセット**
   - 前の音節の証拠（accumulated evidence）を次の音節処理前に消去することが最大の性能改善要因（+15〜23%）
   - 第8ガンマユニット（音節終端の内部推定）がこのリセットをトリガーする

2. **2番目の要因：シータ・ガンマ結合**
   - 外因性・内因性いずれの結合でも性能が向上（+8.6%）
   - 自然な発話速度では外因性vs内因性の差は小さいが、**発話速度が3倍に圧縮された場合**には外因性（刺激駆動）シータの優位性が有意に現れる

3. **BIC（ベイズ情報量規準）**ではバリアントAが最優秀
   - 単純な正解率では区別できないA/Bの差も、モデルの確信度（precision）まで考慮すると外因性シータモデルが優れる

---

## 予測符号化の動作原理

モデルは **Dynamic Expectation Maximisation (DEM)** アルゴリズムを用いて推論を実行します。

1. 内部の音節ユニット＋ガンマユニットが**トップダウン予測スペクトログラム**を生成
2. 実際の入力スペクトログラムとの差分（予測誤差）を計算
3. 予測誤差がヒエラルキーを逆伝播し、各ユニットの活性値を更新
4. 入力と一致する音節ユニットの活性値が上昇、不一致のものは低下
5. 各ガンマシーケンスウィンドウ内で最高活性値を示した音節ユニットを「認識結果」として出力

> "The activity level increases in those syllable units whose spectrogram is consistent with the sensory input and decreases in the others. In the ideal case, online prediction error minimisation leads to elevated activity in one single syllable unit matching the input syllable." [Inference Process](https://alphaxiv.org/abs/2006.10800?page=3)

---

## まとめと意義

| 観点 | 内容 |
|---|---|
| 神経科学的意義 | シータ・ガンマ結合と予測符号化が機能的に協調できることを計算論的に示した |
| 認知科学的意義 | 発話速度が変動する場合、外因性（刺激依存）シータ振動が音節デコードに不可欠になる |
| ASRへの示唆 | 振動ベースのトップダウン更新機構が自動音声認識システムの雑音耐性・可変速度対応に貢献しうる |
| 限界 | 単一シータニューロンでは onset 検出精度が約53%にとどまり、より多くのニューロンを用いることで改善の余地がある |

Precossは、「いつ音節が来るか」という**時間的予測**と「どの音節か」という**内容的予測**を統一的な推論フレームワークに組み込んだ点で、神経科学と計算論的モデリングを橋渡しする重要な貢献です。