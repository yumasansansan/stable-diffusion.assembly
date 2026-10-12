<!--
SPDX-FileCopyrightText: 2026 Yuma Kakei <yumasansansan@gmail.com>
SPDX-License-Identifier: GPL-3.0-or-later

このファイルは README.md の日本語版で，stable-diffusion.assembly の一部です．SPDX の行は，著作権者とライセンスを REUSE 仕様の機械可読な形式で示しています：GNU General Public License バージョン 3 またはそれ以降のバージョン（LICENSES/GPL-3.0-or-later.txt）．
-->

# stable-diffusion.assembly
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp) を CPU で動かすときに時間のかかる演算を，独自のカーネルで計算するプロジェクトです．カーネルは，どのプロセッサにもコンパイラが自動ベクトル化するシンプルな C++ と，x86-64 の命令セットの各レベル向けに手書きしたアセンブリからなり，実行時に選ばれます．

[English](README.md) | 日本語

## 概要

stable-diffusion.cpp は，モデルの計算を [ggml](https://github.com/ggml-org/ggml) の CPU バックエンドで行います．このプロジェクトは stable-diffusion.cpp と ggml をほぼそのままにし，小さなパッチ（`patches/`）を通じて，ggml の CPU バックエンドの重い演算を `sources/` のカーネルに任せます．

- **行列積**（`MUL_MAT`，線形層），**畳み込み**（`CONV_2D`），**attention**（`FLASH_ATTN_EXT`）を，パックしたパネルの行列積として，マイクロカーネルがタイルずつ計算します．スレッドへの仕事の配分は，終わりに近づくほど小さくなるブロック単位で行い，ハイブリッド構成のプロセッサでも速いコアと遅いコアがそろって終わるようにしています．
- 3×3 の畳み込みは，16×16 以上の画像では **Winograd の F(4x4, 3x3)**（乗算 4 分の 1），最も深い層の 8×8 の画像では **F(2x2, 3x3)**（乗算 2.25 分の 1）で計算します．どちらもタイルの少ない画像には専用の経路があります．その他の畳み込みは，im2col のように画像の複製を書き出さず，**直接**計算します．
- **attention** は，クエリのブロックごとに，スコアから出力までをキャッシュ内で完結させます．softmax の指数は，キーの区間ごとに，値との積和の直前に計算します．
- 積の後に続く演算（バイアス，スケール，GroupNorm・LayerNorm とその scale・shift，SiLU，GEGLU のゲート付き GELU，残差接続）は，積に**融合**し，出力がキャッシュにある間に計算します．

各カーネルには 3 つの形態があります．`cxx` はイントリンシクスを使わない素直な C++ で，コンパイラが AVX2・AVX-512・NEON のどれにも同じようにベクトル化します．`assembly` は命令セットの各レベル向けの手書きアセンブリ（`sources/x86_64/`）で，これが既定です．`upstream` はこのプロジェクトのカーネルを使わない，ggml 自身の計算です．形態とレベルは実行時に選ばれます．

## 速度

SD 1.5，512×512，20 ステップ，Intel Core i7-1360P（Golden Cove 4 コア・Gracemont 8 コア，AVX2，電力制限下）での，サンプラーと VAE デコーダーの時間です．

| | サンプリング（20 ステップ） | VAE デコーダー | |
| --- | ---: | ---: | --- |
| stable-diffusion.cpp（Release，`-march=native`，最速のスレッド数 8） | 387.45 秒 | 32.93 秒 | |
| stable-diffusion.assembly（12 スレッド） | 70.8 秒 | 3.16 秒 | 5.5 倍，10.4 倍 |

UNet の時間は，畳み込み（約 37%），線形層（37%），attention（22%）が占めます．線形層は電力制限下でプロセッサの FMA が出せる最大の約 70%，attention は 60% に達しています．畳み込みは Winograd 変換により，直接畳み込みの演算数に換算するとその 1.2〜2.5 倍の速さです．プロファイル（`SDASM_PROFILE=1`）は，演算ごとに形状と段階別の内訳を示します．

このプロセッサ以外での計測は GitHub のランナーで行います（[GitHub のランナーでの計測](#github-のランナーでの計測)）．

## 精度

カーネルは，stable-diffusion.cpp より精度を落としません．ggml は，重みが FP16 の場合（SD 1.5 の線形層と畳み込み）は画像を FP16 に丸めてから積を取り，丸めた数の積を足し合わせます．AVX2 と AVX-512 のカーネルは画像を単精度のまま扱うので，出力は正確な値に 1〜2 桁近くなります．テスト（`tests/ops_test.cc`）は，すべての形態・レベルを倍精度で計算した値と比べます．下表の値は誤差を，足し合わせた積の絶対値の和で割ったものです．

| 演算 | AVX2 / AVX-512 / C++ | AMX | ggml（上流） |
| --- | ---: | ---: | ---: |
| 線形層（FP16 の重み） | 0.4〜2.4 × 10⁻⁷ | 0.5〜1.8 × 10⁻⁶（AMX-BF16），3.5〜7.0 × 10⁻⁸（AMX-FP16） | 2.9〜19 × 10⁻⁵ |
| 畳み込み，F(2x2, 3x3) と直接（FP16 の重み） | 0.7〜2.1 × 10⁻⁷ | 1.2〜6.8 × 10⁻⁶．AMX-FP16 の直接畳み込みは 0.6〜0.8 × 10⁻⁷ | 3.9〜13 × 10⁻⁵ |
| 畳み込み，F(4x4, 3x3)（FP16 の重み，16×16 以上） | 5〜11 × 10⁻⁷ | （AMX では F(2x2, 3x3)） | 4.1〜13 × 10⁻⁵ |
| attention（単精度） | 1.4〜19 × 10⁻⁸（ggml のグラフと同等か，より正確） | 1.3〜7.1 × 10⁻⁸ | 2.0〜16 × 10⁻⁸（グラフ計算） |

**AMX のタイル**は bfloat16（仮数 8 ビット．FP16 は 11 ビット）の積を計算します．そこで各数を x = x1 + x2，w = w1 + w2 の 2 つの部分に分け（それぞれ最近接丸め），積を x1 w1 + x1 w2 + x2 w1 として計算します．こうすると正確な積との差は 2⁻¹⁵ 以内に収まり，ggml の FP16 への丸め（2⁻¹¹）より 16 倍正確です．AMX-FP16 を持つプロセッサ（Granite Rapids）では，FP16 の重みをそのままタイルに載せ，相手の数を FP16 の 2 つの部分（x1 は ggml 自身の丸め，x2 はその残り）に分けます．積は w x1 + w x2 の 2 つで，どちらも正確です．正確な積との差は 2⁻²² 以内で，ggml の丸めより大きくなることはありません．単精度並みの精度で，積の数も 3 分の 2 になります．単精度の 2 つの数どうしの積である Winograd の積は，bfloat16 の 2 分割のままです．ggml が単精度で計算する attention は，タイル上でも単精度で計算します．単精度の数は bfloat16 の 3 つの部分で正確に表せるので，積を x1 y1 + x1 y2 + x2 y1 + x1 y3 + x2 y2 + x3 y1 の 6 つの積として計算します．和が小さいうちに小さい 5 項を先に足し，x1 y1 を最後に足すと，出力は ggml と同等かそれ以上に正確になります（ブロックごとにまとめて足した場合は最大 4 倍誤差が大きくなりました）．GEGLU の GELU は，ggml が FP16 の表から引くのに対し，単精度で計算します．

## 量子化した積：精度を落として速く

`SDASM_TUNE=bits=8` を指定すると，積を整数で計算します．stable-diffusion.cpp と同じ画像ではなくなりますが，同じモデルの画像をより短い時間で生成します．積の 2 つのオペランドの各数を，数ビットの整数とスケールの積に丸め，整数どうしの積を 32 ビットで正確に足し合わせ（VNNI の `vpdpbusd` と同じく，バイトとバイトの積 4 つを 1 レーンに足す），その和をスケールで戻します．

- **重み**は符号付きで，出力チャネルごと，K のグループ（`group`，既定は 64 個）ごとにスケールを持ちます．量子化は 1 度だけで，ステップ間で保持します．
- **入力**はゼロ点付きの符号なしで，行（トークン，画像の位置，Winograd 変換のタイル）ごと，やはり K のグループごとにスケールを持ちます．SiLU や GELU の後の活性はほぼ 0 の片側にあり，対称なスケールでは 1 ビット分を無駄にします．
- 3×3 の畳み込みは **F(2x2, 3x3)** で計算し，変換後のタイルの要素ごとに別のスケールを持たせます（[Andri ら，2022](https://arxiv.org/abs/2209.12982) が F(4x4, 3x3) を量子化した *tap-wise* の方法）．整数の F(4x4, 3x3) は丸めの影響が F(2x2) の 2.3 倍になり，直接畳み込みは 1.3 分の 1 ですが積が 2.25 倍になります．それ以外の畳み込みは直接計算します．
- **self-attention** は，クエリとキーのスコアをヘッドの次元全体で，値と指数の積和をキーのグループごとに量子化します（指数はクエリごと）．全キーで 1 つのスケールにすると，8 ビットの指数では小さい値が丸めで消えてしまうためです．プロンプトの 77 トークンに対する **cross-attention** は単精度のままです．計算量は 50 分の 1 で，丸めの影響は 10 倍になるからです．
- 線形層は，後に続く GEGLU のゲート付き GELU と融合したまま計算します．256 行未満の線形層（テキストエンコーダ，プロンプトを cross-attention のキーと値に射影する層，時刻埋め込み）は単精度のままです．計算量は数百分の 1 ですが，絵全体を左右するためです．

マイクロカーネルは，単精度のものを整数にしたものです．AVX-VNNI 用の 6×16（`x86_64/quant_avx2.S`．Alder Lake 以降のコアが持つ VEX 版 `vpdpbusd`），AVX512-VNNI 用の 6×16 と 6×32（`quant_avx512.S`），そして整数演算をコンパイラがベクトル化する C++ 版です．16 列のパネルの量子化は AVX2 で行います（`quantize_avx2.S`）．どれも同じ整数を計算するので，attention の指数を除いて出力はビット単位で一致します（`tests/ops_test.cc quantized`）．

i7-1360P では，`vpdpbusd` により P コアは FMA の 4 倍，E コアは 2 倍の積を計算します．SD 1.5，512×512，20 ステップ：

| | サンプリング（20 ステップ） | VAE デコーダ |
| --- | ---: | ---: |
| 単精度（既定） | 71.3 秒 | 3.18 秒 |
| `bits=8` | 38.7 秒 | 2.31 秒 |

SD 1.5 自身の活性で測ると，各演算の出力と正確な値との差（L2 ノルムの相対値）は，線形層で 5 × 10⁻³，畳み込みで 6〜7 × 10⁻³，self-attention で 1〜4 × 10⁻³ です．画像の品質は同等ですが，既定のサンプラー Euler A は各ステップでノイズを加えるため，この差が 20 ステップを通じて持ち越されます．8 つのプロンプトのうち 7 つは細部が動いた同じ絵に，1 つは別の絵になりました．正確な画像との PSNR は中央値 25.8 dB（17.9〜34.2），SSIM は 0.90（0.77〜0.98），`group=32` では 27.5 dB と 0.92 です．stable-diffusion.cpp 自身の 8 ビット量子化モデル（`--type q8_0`．線形層のみ，32 個ごとのグループの W8A8）でも画像は動き，最初のプロンプトで 34 dB（`bits=8` も 34 dB），`bits=8` が別の絵にしたプロンプトで 23 dB 離れます．

| チューニング | |
| --- | --- |
| `bits=8\|6\|4` | 重みのビット数．入力は `input_bits`（8）のまま |
| `input_bits=2..8` | 入力のビット数 |
| `group=32\|64\|128` | スケールを共有する K の数．32 は正確な画像により近く，128 はここでは 64 より速くなりません |
| `weight_group=0` | 重みのスケールをグループごとではなく K 全体で 1 つにする |
| `quantize=1..7` | 量子化する演算．1 は線形層，2 は畳み込み，4 は attention で，足し合わせて指定（既定は 7，すべて） |

重みを 4 ビットや 6 ビットにしても，整数は 1 バイトのままで積の数も同じなので，時間は縮まらず画像だけが劣化します．

## 命令セットのレベル

| レベル | 拡張 | 手書きのカーネルでの用途 |
| --- | --- | --- |
| `avx2` | AVX2，FMA，F16C（x86-64-v3） | マイクロカーネル（6×16），softmax の指数，パッキング，F(2x2, 3x3) と F(4x4, 3x3) の変換，ゲート付き GELU |
| `avx512` | AVX-512 F，VL，BW，DQ（AVX10.1） | 同じものを 16 レーン幅で．画像の端ではマスク付きロード．Intel のプロセッサでは 6×32 のマイクロカーネル（16 レーンの FMA を毎サイクル 2 つ実行するのにロードが追いつく形） |
| `amx` | AMX-TILE，AMX-BF16，AVX512-BF16．あれば AMX-FP16 | 線形層と畳み込み（Winograd を含む）の積を，2 分割した bfloat16 でタイル上で計算（AMX-FP16 があれば，重みはそのまま，相手を FP16 で 2 分割）．attention は bfloat16 の 3 分割．Winograd の重み変換は AVX-512 で．他は `avx512` と同じ |

使われるレベルは，プロセッサが持つ最も豊かなものです．`SDASM_ISA=avx2|avx512|amx` でそれより下のレベルに抑えられます．テストは，プロセッサが持つすべてのレベルで各演算を計算します．x86-64-v3 向けのビルド（`SDASM_ARCH=avx2`，配布するビルド）は全レベルを実行でき，その場合 C++ 部分は AVX2 向けにコンパイルされます．

AVX-512 のその他の拡張は，この精度のままではこれらの演算を速くできません．
- VNNI（と AVX-VNNI）は整数の積で，使うには数を量子化しなければなりません．上の量子化した積で使います．
- AVX512-BF16 の `vdpbf16ps` は FMA の 16 に対して 32 の積を計算しますが，2 分割のため元の積 1 つにつき 3 つの積が要り，単精度の FMA の 1.5 倍の命令数になります．llvm-mca の Cooper Lake・Sapphire Rapids・Zen 4・Zen 5 のモデルでは速くなりません．実機で計測する変種として残しています（`SDASM_TUNE=bf16=1`，パッキングはタイルと共用）．
- AVX512-FP16 は FP16 で和を取るため，途中の和がすべて 11 ビットに丸められます．
- IFMA，VBMI，VBMI2，BITALG，VPOPCNTDQ，VP2INTERSECT，GFNI，VAES，VPCLMULQDQ は単精度の計算と関係がありません．

AMX のタイルは別で，AVX-512 の FMA の 16 倍の積を計算するため，元の積 1 つに 3 つの積を使っても 5 倍になります．

## ビルド

Linux の x86-64 で（他は当面 C++ のカーネルのみ），[apt.llvm.org](https://apt.llvm.org) の Clang・LLD・LLVM ツール（バージョン 23），CMake 3.29 以上，Ninja でビルドします．`ci/setup.sh` は CI と同じく Ubuntu にこれらを入れます．

```bash
git clone --recursive https://github.com/yumasansansan/stable-diffusion.assembly.git
```

```bash
cmake --preset release
```

```bash
cmake --build --preset release
```

プリセット（`cmake --list-presets`）は，`release`（-O3，ThinLTO，既定は x86-64-v3 向け），`debug`，`relwithdebinfo`，`sanitize` と `tsan`（address と undefined，thread と undefined），そして Intel SDE 用に x86-64-v3 でテストを作る `emulated` です．`-DSDASM_ARCH=native` は，C++ をビルドしたマシン向けにコンパイルします．

プログラムは stable-diffusion.cpp の `build/release/bin/sd-cli` で，オプションはすべてそのまま使えます．

```bash
build/release/bin/sd-cli -m v1-5-pruned-emaonly.safetensors -p "a lovely cat" --steps 20 -t 12
```

### 環境変数

| 変数 | |
| --- | --- |
| `SDASM_KERNELS=assembly\|cxx\|upstream` | カーネルの形態（既定は assembly） |
| `SDASM_ISA=avx2\|avx512\|amx` | 手書きのカーネルが使う最も豊かなレベル |
| `SDASM_TUNE=name=value,...` | ブロックの大きさやカーネルの選択（`sources/kernels.h` の `Tuning`）．`bits=8` は量子化した積，`wide=0\|1` は AVX-512 の 6×32 マイクロカーネル，`bf16=1` は AVX512-BF16 の積，`amx_fp16=0` は AMX-FP16 を持つプロセッサで AMX-BF16 の形式を使う，`amx_attention=0` は attention をベクトルユニットで計算 |
| `SDASM_PROFILE=1` | 終了時に，各演算の時間を形状別・段階別に表示 |
| `SDASM_STORE_MB=n` | パックした重みを n MiB までステップ間で保持．量子化した重みは，指定がなければ 8 GiB まで保持 |

## テスト

テストとベンチマークは `-DSDASM_BUILD_TESTS=ON -DSDASM_BUILD_TOOLS=ON` でビルドします（`ci/build.sh release avx2` はこの設定で構成し，ツールチェーンを確かめてからビルドします）．

```bash
cmake --preset release -DSDASM_BUILD_TESTS=ON -DSDASM_BUILD_TOOLS=ON
```

```bash
cmake --build --preset release
```

```bash
ctest --preset release
```

`sdasm_ops_test <演算> [スレッド数]` は，各演算をすべての形態・レベルで倍精度と比べ，それぞれの誤差を表示します．`ci/test.sh` はそれを全演算について実行します．`ci/emulated.sh <sde64>` は，`emulated` プリセットのテストを [Intel SDE](https://www.intel.com/content/www/us/en/developer/articles/tool/software-development-emulator.html) 上で，Haswell・Skylake-SP・Ice Lake-SP・Sapphire Rapids・Arrow Lake・Granite Rapids として（AMX のタイルも含めて）実行します．

`sdasm_bench` は SD 1.5 の各演算を実際の形状で，各形態（`--forms assembly,avx2,avx512,amx,cxx,upstream`）と変種（`--variant label:isa=avx512,wide=0`）について計時します．`--peak` は，各幅でプロセッサが出せる FMA の最大を計ります．

## GitHub のランナーでの計測

ワークフロー `Measure`（`.github/workflows/measure.yml`）は手動で起動します．一度ビルドし，そのビルドを指定した数（既定 60）のランナーで同時にテスト・計時します．ランナーは GitHub が持つ様々なプロセッサ上の仮想マシンで，AMD の Zen 3・4・5，Intel の Ice Lake・Emerald Rapids・Granite Rapids が確認されており，AVX-512 や AMX を持つものも含まれます．各ランナーは自分のプロセッサを記録し，実行のサマリーではランナーをプロセッサごとにまとめます．プロセッサごとに，全演算・全形態・全変種の速度の中央値，レベル間の速度比，FMA の最大に対する割合を示します．

## ライセンス

GPL-3.0-or-later（`LICENSES/GPL-3.0-or-later.txt`）．パッチは，変更先のコード（MIT License）の著作権者も記しています．サブモジュールは，それぞれのプロジェクトのライセンスに従います．すべてのファイルは [REUSE](https://reuse.software) 仕様の形式で著作権者とライセンスを記しており，CI の `reuse lint` が確かめます．
