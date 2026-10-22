# 可复现性与工程规范：环境契约与证据链

> 本文档补充 `docs/design_notes.md`（固定设计，不得修改）。它不改变任何
> 预注册假设、FDR 族、SESOI 或能力门禁，只说明**结果目录如何证明自己是在
> 同一个软件/数值环境里产生的**，以及**历史结果如何避免被后续运行静默
> 重新认证**。

## 0. 已有保证与剩余缺口

仓库已经固定了代码内容哈希、配置、训练种子、数据流指纹、跨进程重放和
累计 `artifact_index.json`。这些读数回答"同一份代码、同一份配置、同一份
数据是否给出同一条轨迹"。仍有两个工程缺口会让"可复现"名不副实：

1. **环境漂移**：`torch.__version__` 只钉住框架本身，NumPy、BLAS wheel、
   `pyproject.toml`/`requirements-ci.txt` 的声明同样会改变数值路径。把不同
   环境下产生的记录追加到同一个 `results/`，索引仍然会全部认证，却不再是
   同一个数值配方。
2. **追加时的静默重认证**：`artifact_index.json` 在每次运行结束时才重新
   计算哈希。如果两次运行之间有人编辑或截断了一个已被认证的 JSONL 文件，
   下一次运行可能直接向它追加，并在结束时把修改后的内容写成新的"已验证"
   条目。

## 1. 环境记录（`environment.json`）

每次正式运行把 `cage.experiments._environment_record` 写入
`environment.json`（同时进入 manifest 与每条记录的 provenance stamp）。
除了已有的 Python/torch/平台/线程/BLAS 环境变量/确定性开关外，现在还包括：

```text
python_full_version      完整解释器版本串（短版本不含构建信息）
package_inventory        已安装发行包 name -> version（稳定排序）
dependency_manifest      pyproject.toml / requirements-ci.txt 等声明文件的
                         {bytes, sha256}
environment_sha256       对全部非易失字段做规范化 JSON 后的 SHA-256 前 16 位
```

`environment_sha256` 覆盖非易失字段，因此新增一个环境探针字段会自动进入
指纹，而不是被静默忽略。`argv`、`cwd`、`repo_root`、`git_dirty` 与
`pythonhashseed` 是**启动元数据**，记录在案但不参与指纹：从另一个目录启动
同一次实验仍是同一个环境。

## 2. 追加运行的环境兼容性（`environment_history.jsonl`）

每次运行在写入新的 `environment.json` **之前**读取旧的记录，比较后向
`environment_history.jsonl` 追加一行：

```text
run_id / setup_sha256 / git_sha / created_utc
previous_environment_sha256 / current_environment_sha256
compatible / identical / reason / differences
```

- `compatible=true`：关键软件/数值字段一致；`identical=true` 表示连
  `platform`、CPU 型号等 advisory 字段也一致。
- `compatible=false`：至少一个关键字段变化（torch/Python/包清单/线程数/
  确定性开关/TF32/BLAS 环境/声明依赖哈希）。运行默认继续并打印警告，但
  这个 verdict 会留在证据链里。
- `--strict-environment`：关键字段漂移时**不开始训练**，以 exit 3 结束，
  并在 `run_index.jsonl` 写 `aborted` 事件。
- 首次运行或旧记录损坏时不会伪造一个"匹配"：前者是 `compatible=null`，
  后者是 `compatible=false` 且 `reason` 说明记录不可读。

advisory 字段（平台、CPU 型号、核数、可执行文件路径、CUDA 设备描述等）变化
只说明"同一软件配方的另一台机器"，不阻止运行；它们仍写进差异列表，供报告
限定"同机同构建"的解释边界。

## 3. 只读环境检查

```bash
make verify-env OUT=results
# 等价于
python3 -m cage.experiments --check-environment --out results
```

读取 `results/environment.json`（缺失时回退到 `<tag>_manifest.json`），
与当前进程重建的环境比较：

- exit 0：关键字段一致；
- exit 1：关键字段漂移，逐项打印 `[critical] field: recorded=... current=...`
  与嵌套包清单的 missing/extra/changed keys；
- exit 2：没有可用的环境记录或文件损坏。

该命令**不写入任何文件**，可在分析前、复现实验前或评审时反复运行。

## 4. 追加式证据链

`cage.experiments._write_result`（以及 `run_index.jsonl` 的写入）在追加前
检查目标文件是否已被 `artifact_index.json` 认证：

- 已认证且当前 size/SHA-256 与条目一致 → 允许追加；
- 已认证但内容被编辑/截断 → 抛 `TrackingError`，拒绝追加；必须先运行
  `python3 -m cage.experiments --verify-artifacts --out results` 并恢复文件；
- 已认证但文件缺失 → 同样拒绝，不允许下一次运行用新内容静默补回一个
  "看起来完整"的历史；需要显式重建结果目录；
- 索引本身损坏 → 同样拒绝，避免"读不出旧账本就当作没有旧账本"；
- 新文件/未进入索引的历史文件 → 允许追加，本次结束时会纳入索引。

因此 `artifact_index.json` 仍然是累计清单，但不再能把一次被篡改的历史
"洗白"成新的认证条目。运行结束后的完整校验：

```bash
make verify-artifacts OUT=results
python3 -m cage.experiments --verify-artifacts --out results
```

## 5. 解释边界

- 环境一致 + 代码/配置/数据一致，才构成"可重建"的充分记录；本仓库仍不
  承诺跨硬件/指令集/内核库的 bit-exact 复现（见 `cage/repro.py` 的 scope）。
- `package_inventory` 是进程可见的安装集合，不记录系统共享库（如 cuBLAS
  的具体动态库路径）；`torch` 构建号、CUDA runtime、cuDNN 版本与
  `numeric_env` 是这部分边界的补充读数。
- 环境历史是**伴随证据**，不改变任何统计判定；它只让"这条结果是在哪个
  环境里产生的"从隐含假设变成可检查的记录。
- 追加保护检测的是"已被索引认证的内容是否被改动"，不是密码学签名：能同时
  改写结果文件与索引的人仍可伪造整条链。它防止的是误操作、截断和选择性
  追加，不是恶意攻击者。

## 6. 复现

```bash
python3 -m unittest tests.test_reproducibility_contract -v
python3 -m unittest tests.test_artifact_append_guard -v
python3 -m cage.experiments --only sanity --quick --out results_quick
make verify-env OUT=results_quick
make verify-artifacts OUT=results_quick
make check
```
