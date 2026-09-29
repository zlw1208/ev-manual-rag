# EV Manual RAG

面向新能源汽车用户手册的中文智能问答系统。项目计划支持多车型过滤、混合检索、答案引用、安全拒答和离线评测。

## 当前状态

项目已完成可演示 MVP：包括说明书下载与解析、结构化分块、本地混合检索、Cross-Encoder
重排、Qwen Plus 带引用回答、FastAPI 问答接口、Streamlit 用户页面及离线评测。

## 目录结构

```text
app/
├── api/          # FastAPI 接口
├── ingestion/    # 下载、解析、清洗和分块
├── retrieval/    # 向量检索、关键词检索和重排序
├── generation/   # Prompt、答案生成、引用和拒答
└── evaluation/   # 检索与回答质量评测
data/
├── raw/          # 原始 PDF，仅保存在本地
├── processed/    # 解析后的结构化数据，仅保存在本地
└── eval/         # 人工评测问题集
frontend/         # 演示界面
scripts/          # 数据处理和运维脚本
tests/            # 自动化测试
docs/             # 架构和实验记录
```

## 本地运行

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
uvicorn app.api.main:app --reload --port 8001
```

打开 `http://127.0.0.1:8001/docs` 查看接口文档，健康检查地址为 `http://127.0.0.1:8001/health`。

另开一个终端启动 Streamlit 用户问答页：

```powershell
python -m pip install -e ".[dev,retrieval,generation,ui]"
streamlit run frontend/streamlit_app.py
```

浏览器打开 `http://localhost:8501`。页面支持对话式提问、说明书引用、检索原文与
Reranker 分数展示。前端默认调用 `http://127.0.0.1:8001/ask`，也可以通过
`RAG_API_BASE_URL` 环境变量切换后端地址；当前版本不包含 PDF 上传和知识库管理。

`POST /ask` 请求示例：

```json
{
  "question": "冰雪路面使用防滑链时应该装在哪些车轮？"
}
```

响应包含生成答案、说明书引用、Top-K 检索证据、首阶段与 Reranker 分数、耗时和 Token 用量。

## 请求追踪与日志

FastAPI 为每个请求生成 32 位 `request_id`，通过响应体和 `X-Request-ID` 响应头返回。
Streamlit 会在答案下方显示请求编号，可用它关联后端的两类结构化 JSON 日志：

- `http_request_completed`：请求方法、路径、HTTP 状态码和总耗时；
- `qa_completed`：是否拒答、证据数量、生成耗时和 Token 用量。

异常事件只记录异常类型。为避免用户隐私进入日志，系统不记录完整问题和答案，只记录问题长度及
SHA-256 短指纹；API Key 同样不会写入日志。

## Docker Compose 部署

容器化版本使用独立的 Qdrant 服务，完整链路为：

```text
Streamlit → FastAPI → Qdrant
                         ↓
                  qdrant_data Volume
```

确保项目根目录存在本地 `.env`，其中配置 `DASHSCOPE_API_KEY` 等生成模型参数，然后运行：

```powershell
docker compose up --build
```

Compose 会依次启动：

1. `qdrant`：向量数据库服务，宿主机端口为 `6333`；
2. `indexer`：一次性索引初始化任务，集合非空时跳过重建；
3. `api`：FastAPI 服务，地址为 `http://localhost:8001`；
4. `frontend`：Streamlit 页面，地址为 `http://localhost:8501`。

三个宿主机端口都只绑定 `127.0.0.1`，不会直接暴露给局域网。Qdrant 管理页面位于
`http://localhost:6333/dashboard`；前端会等待 API 健康检查通过后再启动。

向量数据保存在 Docker 命名 Volume `qdrant_data` 中，模型缓存保存在 `model_cache` 中。
普通停止和重建容器不会删除数据：

```powershell
docker compose down
docker compose up
```

更新说明书分片后，可显式重建远程集合：

```powershell
docker compose run --rm indexer python scripts/build_index.py `
  --qdrant-url http://qdrant:6333 --wait-seconds 60
```

查看容器内 Qdrant 的集合与样例数据：

```powershell
python scripts/inspect_index.py --qdrant-url http://localhost:6333 --limit 3
```

只有执行 `docker compose down -v` 才会删除两个命名 Volume，其中包括向量数据。`.env` 已被
`.dockerignore` 排除，不会写入镜像；Compose 只在运行时将其注入 `api` 容器。

运行测试：

```powershell
pytest
```

## 下载与解析手册

数据源统一维护在 `data/sources.json`。下载首份已验证的问界 M5 纯电版手册：

```powershell
python scripts/download_manuals.py aito-m5-ev
```

将 PDF 解析为保留页码、可见标题、安全标识和正文的 JSONL：

```powershell
python scripts/parse_manual.py data/raw/aito-m5-ev.pdf --document-id aito-m5-ev
```

输出文件位于 `data/processed/aito-m5-ev.pages.jsonl`。解析器会检测单栏与双栏页面，且不依赖可能损坏的 PDF 书签。

按照章节、自然段和列表边界生成检索分片：

```powershell
python scripts/chunk_manual.py data/processed/aito-m5-ev.pages.jsonl --source-id aito-m5-ev
```

默认分片目标长度为 450 个中文字符，硬上限为 800；只有超长段落被强制拆分时才使用 80 字重叠。目录分片会被保留用于审计，但设置为不可检索。

## 建立与查询混合索引

安装检索依赖并建立本地 Qdrant 索引：

```powershell
python -m pip install -e ".[dev,retrieval]"
python scripts/build_index.py
```

首次构建会下载中文向量模型；模型默认缓存在项目的 `.cache/fastembed`，索引保存在 `qdrant_storage`，二者都不会提交到 Git。后续构建和查询会直接使用本地缓存。

执行 Dense、Sparse 或混合检索：

```powershell
python scripts/search_manual.py "车充不进去怎么办" --mode dense --model M5 --power-type 纯电
python scripts/search_manual.py "BMS故障" --mode sparse --model M5 --power-type 纯电
python scripts/search_manual.py "预约充电为什么没有启动" --mode hybrid --model M5 --power-type 纯电
```

对 Hybrid 召回的前 10 个候选使用多语言 Cross-Encoder 二次排序：

```powershell
python scripts/search_manual.py "低压电池没电怎么办" --mode hybrid --rerank --limit 5
```

查看本地向量数据库的集合状态、样例数据和向量摘要：

```powershell
python scripts/inspect_index.py --limit 3
```

添加 `--show-vector` 可以打印完整向量，通常只在调试时使用。

CPU 基线使用 `BAAI/bge-small-zh-v1.5` 生成中文语义向量；关键词检索使用 `jieba` 分词和 BM25 风格稀疏权重，最终由 Qdrant RRF 融合。Dense 模型通过命令行参数配置，后续可以切换为 BGE-M3。

## 数据使用原则

- 数据源只选择汽车厂商官方网站公开提供的用户手册。
- 原始手册及完整解析文本不提交到 Git 仓库。
- 仓库只保存来源地址、导入代码和少量评测数据。
- 所有回答必须附带原始手册引用；涉及制动、高压电池、碰撞等高风险问题时优先拒答或建议联系官方售后。

## MVP 路线

1. 建立官方手册数据源清单并下载首批 PDF。（已完成首份）
2. 提取标题层级、正文、页码、警告和提示信息。（已完成页级版本）
3. 实现结构化分块。（已完成第一版）
4. 建立本地向量及关键词索引。（已完成）
5. 实现车型过滤和混合检索。（已完成第一版）
6. 实现 Cross-Encoder Reranker。（已完成第一版）
7. 实现带引用回答和无依据拒答。（已完成）
8. 构建测试问题集，评估 Recall@K、MRR、回答要点、引用质量和响应延迟。（已完成第一版）

## 检索评测

第一版人工标注问题集位于 `data/eval/retrieval_questions.jsonl`，每个问题绑定一个或多个正确的 `chunk_id`。运行 Dense、Sparse 和 Hybrid 基线评测：

```powershell
python scripts/evaluate_retrieval.py
```

保存每个问题的排名、延迟和 Top-5 结果，便于后续与 Reranker 对比：

```powershell
python scripts/evaluate_retrieval.py --output data/eval/baseline_results.json
```

评测 Hybrid 加 Reranker 的效果：

```powershell
python scripts/evaluate_retrieval.py --modes hybrid hybrid_rerank
```

当前 30 题 CPU 基线（Reranker 候选数为 10）：

| 检索方式 | Recall@1 | Recall@3 | Recall@5 | MRR@5 | 平均延迟 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Hybrid | 86.7% | 96.7% | 100.0% | 0.919 | 51.9 ms |
| Hybrid + Reranker | 96.7% | 100.0% | 100.0% | 0.978 | 868.7 ms |

Reranker 使用 Apache-2.0 许可的 `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`，加载约 118 MB 的 AVX2 量化 ONNX 权重。详细逐题结果保存在 `data/eval/reranker_results.json`。

为避免 Cross-Encoder 截断长分片尾部的信息，重排阶段会对超过 500 字的候选分片进行子段化：

- 每个子段最多 400 字，相邻子段重叠 80 字；
- Reranker 分别计算查询与每个子段的相关性；
- 使用 Max Pooling（取最高子段分数）作为原分片的最终重排分数；
- 排序后仍将完整原分片交给生成模型，引用的分片 ID 和页码保持不变。

重排结果会记录最佳子段、子段序号和子段数量，便于定位分数来自原分片的哪个位置。

## 生成带引用的回答

安装回答生成依赖，并配置阿里云百炼的 `qwen-plus`：

```powershell
python -m pip install -e ".[dev,retrieval,generation]"
$env:DASHSCOPE_API_KEY="你的百炼 API Key"
$env:LLM_MODEL="qwen-plus"
$env:LLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
```

也可以复制 `.env.example` 为 `.env` 后填写配置；程序会从项目根目录自动加载 `.env`。
`.env` 已加入 `.gitignore`，不得将真实 API Key 提交到仓库。

先检查检索结果和即将发送给模型的 Prompt，不会调用模型 API：

```powershell
python scripts/ask_manual.py "低压蓄电池没电怎么办" --dry-run
```

生成经过引用校验的回答：

```powershell
python scripts/ask_manual.py "低压蓄电池没电怎么办"
```

程序通过百炼 OpenAI 兼容的 Chat Completions 接口调用模型，并使用 `qwen-plus` 的
JSON Schema 结构化输出；客户端还会在本地进行二次 Schema 校验，异常时自动重试一次。
模型只返回资料编号，章节、页码和 `chunk_id` 由服务端从检索结果回填；引用不存在或
缺失时，系统拒绝输出未验证答案。

## 端到端问答评测

评测集包含 30 道说明书问答题和 5 道无依据拒答题。运行完整评测：

```powershell
python scripts/evaluate_generation.py
```

首次验证时可以只运行前 3 题：

```powershell
python scripts/evaluate_generation.py --limit 3
```

报告写入 `data/eval/generation_results.json`，包含答案要点召回率、完整答案率、引用命中率、
引用精确率、拒答准确率、端到端延迟、Token 使用量和逐题回答。关键词要点指标用于稳定的
自动回归测试，最终对外报告结果前仍应抽样进行人工事实核查。

仅更新评测标注后，可复用已有回答重新评分，避免重复调用模型：

```powershell
python scripts/evaluate_generation.py --reuse data/eval/generation_results.json
```

当前 35 题基线（30 道可回答题、5 道拒答题）：

| 指标 | 结果 |
| --- | ---: |
| 端到端通过率 | 100.0% |
| 答案要点召回率 | 100.0% |
| 引用命中率 | 100.0% |
| 引用精确率 | 97.2% |
| 拒答准确率 | 100.0% |
| 平均 / P95 延迟 | 3.05s / 5.43s |
| 总 Token | 67,003 |

该结果是当前小规模人工标注集上的开发基线，不代表开放域或所有用户问题的泛化效果。

独立冻结测试集包含 16 道新章节问答和 4 道新拒答题。首次运行前已固定标注，报告会记录
数据集 SHA-256，测试后不依据模型输出修改评分规则：

```powershell
python scripts/evaluate_generation.py `
  --questions data/eval/generation_test_questions.jsonl `
  --output data/eval/generation_test_results.json
```

冻结测试集首次运行结果：

| 指标 | 结果 |
| --- | ---: |
| 自动规则通过率 | 75.0% |
| 答案要点召回率 | 84.4% |
| 引用命中率 / 精确率 | 93.8% / 93.8% |
| 拒答准确率 | 100.0% |
| 平均 / P95 延迟 | 2.95s / 3.63s |
| 总 Token | 34,925 |

测试集 SHA-256 为
`6411f169a6a36b048a4baa9a6fb9ff4df05dee21da6569eeafed9789467be65b`。人工复核发现，
5 个自动失败中有 4 个是正确答案未命中严格关键词，另 1 个是防滑链问题的真实检索遗漏；
因此保留原始 75% 自动分数，同时将关键词评分局限与真实失败分开分析。

加入“长分片子段化 + Max Pooling”后，在不修改冻结测试题及标注的前提下再次运行：

```powershell
python scripts/evaluate_generation.py `
  --questions data/eval/generation_test_questions.jsonl `
  --output data/eval/generation_test_results_passage_rerank.json
```

| 指标 | 首次基线 | 子段化重排 |
| --- | ---: | ---: |
| 自动规则通过率 | 75.0% | 85.0% |
| 答案要点召回率 | 84.4% | 91.7% |
| 引用命中率 / 精确率 | 93.8% / 93.8% | 100.0% / 100.0% |
| 拒答准确率 | 100.0% | 100.0% |
| 平均 / P95 延迟 | 2.95s / 3.63s | 2.95s / 3.81s |
| 总 Token | 34,925 | 31,734 |

其中，原先因答案位于 799 字分片尾部而被重排到第 4 名的“防滑链”正确分片提升到第 1 名，
问题由拒答变为正确回答。剩余 3 个自动失败均引用了正确分片：OTA 答案未复述“4G”，冷却液答案未提及
“泄漏”，另有两处语义正确但未命中严格同义词的情况。因此该结果仍应同时结合人工复核解读。

### 最终独立验收

最终验收集包含 20 道全新的说明书问答题和 5 道无依据拒答题，与开发集和上一轮冻结
测试集分离。题目与标注在运行前冻结，SHA-256 为
`b0dfe019a38d2e42769e3334da52aa749c8f607c669a3956a45704c4c15850eb`。正式验收只运行一次，
并通过 Docker 中的 FastAPI 服务测试“前端之外的完整问答链路”，没有直接调用本地问答对象：

```powershell
python scripts/evaluate_generation.py `
  --questions data/eval/generation_acceptance_questions.jsonl `
  --api-url http://127.0.0.1:8001 `
  --output data/eval/generation_acceptance_results.json
```

| 指标 | 最终验收结果 |
| --- | ---: |
| 自动严格通过率 | 76.0%（19/25） |
| 答案要点召回率 | 89.2% |
| 完整答案率 | 70.0% |
| 引用命中率 / 精确率 | 100.0% / 98.3% |
| 拒答准确率 | 100.0%（5/5） |
| 平均 / P95 延迟 | 3.51s / 4.87s |
| 总 Token | 43,039 |

逐题人工事实复核后，6 个自动失败中有 5 个实际回答完整且引用正确，只是严格字符串匹配
未覆盖“踩下刹车踏板”“不可以”“切勿”“不会执行”等等价表达；另 1 题虽然回答了每月充满和
每周充满，但确实遗漏“长期停放时每月至少使用一次”。因此人工复核正确率为 96.0%（24/25），
同时保留 76.0% 的原始自动分数，不修改冻结题集、标注或正式报告。这个差异说明基于关键词的
自动评测可重复、成本低，但需要配合人工事实核查，后续可增加同义词归一化或 LLM-as-a-Judge，
并继续保留人工抽查以控制误判。

#### 验收后缺陷修复

首次验收报告作为历史基线保持不变。随后修复了两类缺陷：

- 关键词评分器增加否定同义词归一化、有限插入词及局部语序匹配，同时保护肯定/否定极性和数值，
  避免把“可以”和“不可以”或不同阈值判成相同。复用首次回答重新评分后，自动通过率从 76.0%
  提升到 96.0%（24/25），与人工复核一致；结果保存在
  `data/eval/generation_acceptance_results_rescored.json`。
- 对包含“又、分别、哪些、以及”等特征的复合问题增加一次答案完整性复核，并从问题中提取时间、
  阈值和挡位条件，要求检查同一条件下的全部并列动作。普通问题仍只调用一次模型。针对唯一真实
  漏答题进行回归后，答案已同时覆盖“每月至少使用一次车辆”“每月至少充满一次动力电池”和
  “日常建议每周至少充满一次”。

此外，Dockerfile 已启用 pip BuildKit 缓存并延长下载超时，降低弱网络环境下镜像重建失败后重复
下载全部依赖的成本。

修复完成后重新构建 Docker 镜像，并使用同一冻结题集通过 FastAPI 全链路执行回归测试。结果写入
`data/eval/generation_acceptance_results_post_fix.json`，首次验收报告仍保留作为修复前基线：

| 指标 | 修复前正式验收 | 修复后回归 |
| --- | ---: | ---: |
| 自动通过率 | 76.0% | 100.0% |
| 答案要点召回率 | 89.2% | 100.0% |
| 完整答案率 | 70.0% | 100.0% |
| 引用命中率 / 精确率 | 100.0% / 98.3% | 100.0% / 100.0% |
| 拒答准确率 | 100.0% | 100.0% |
| 平均 / P95 延迟 | 3.51s / 4.87s | 4.91s / 7.10s |
| 总 Token | 43,039 | 72,466 |

完整性复核使平均延迟增加 39.8%、P95 延迟增加 45.8%、总 Token 增加 68.4%。因此它只在规则识别为
复合问题且初稿不是拒答时启用；该回归集有 15 道问答实际进入第二次模型复核。结果说明修复提高了
当前冻结集上的完整性，但不能据此宣称所有开放问题均达到 100%，后续仍需持续加入新的未见问题。
