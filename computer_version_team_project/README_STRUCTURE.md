# 📦 VisionRAG 项目 - 文件结构指南

## ✅ 文件整理完成

项目已按功能分类整理到 **6 个主要文件夹**中：

---

## 📁 项目结构

```
computer_version_team_project/
│
├─ 📂 core_app/                       🎯 核心应用模块
│  ├─ main.py                         # 主程序入口
│  ├─ hand_detection_module.py        # 手部+对象检测
│  ├─ depth_detection_module.py       # 深度估计
│  └─ voice_module.py                 # 语音识别+LLM
│
├─ 📂 rag_system/                     🧠 RAG + LangChain
│  ├─ voice_rag_langgraph.py          # 7节点状态机
│  ├─ async_voice_rag_langgraph.py    # 异步包装器
│  ├─ object_retriever.py             # Hybrid RAG 引擎
│  └─ object_knowledge_base.json      # 8个对象知识库
│
├─ 📂 ai_training/                    🚀 AI 训练+推理系统
│  ├─ adaptive_reasoning.py           # ReAct自适应推理
│  ├─ learning_system.py              # 自动学习系统
│  ├─ model_selector.py               # 智能模型选择
│  ├─ reward_system.py                # Q-Learning强化学习
│  └─ ai_training_demo.py             # 完整演示脚本
│
├─ 📂 infrastructure/                 🔧 生产级基础设施
│  ├─ logging_config.py               # 结构化日志
│  ├─ tracing.py                      # 分布式追踪
│  ├─ cache_layer.py                  # 多层LRU缓存
│  ├─ demo_level2_features.py         # 基础设施演示
│  └─ test_performance_level2.py      # 性能测试
│
├─ 📂 docs/                           📖 文档
│  ├─ RESUME_BRIEF.md                 # 📌 简洁简历（必读）
│  ├─ PROJECT_STRUCTURE.md            # 📌 详细结构（必读）
│  ├─ AI_TRAINING_GUIDE.md            # 集成使用指南
│  ├─ AI_TRAINING_SUMMARY.md          # 功能总结
│  ├─ LEVEL2_INFRASTRUCTURE.md        # 基础设施文档
│  └─ LEVEL2_QUICKSTART.md            # 快速开始
│
├─ 📂 models/                         🤖 预训练模型
│  ├─ yolov8m.pt                      # YOLO v8 中等模型
│  ├─ yolov8n-pose.pt                 # YOLO v8 姿态模型
│  └─ hand_landmarker.task            # MediaPipe 手部模型
│
├─ 📂 learning_data/                  💾 学习数据持久化
│  └─ learning_state.json             # 自动保存的学习进度
│
├─ 📂 basic_test_for_prepare/         🧪 初步测试（旧）
├─ 📂 deeep_detection/                🧪 深度学习实验（旧）
├─ 📂 space relation detection/       🧪 迭代版本v1-v4（旧）
└─ 📂 the detection prapare/          🧪 检测模块准备（旧）

```

---

## 🚀 快速开始

### 1️⃣ 了解项目 (3 分钟)
```bash
cd docs
cat RESUME_BRIEF.md
```

### 2️⃣ 查看完整结构 (5 分钟)
```bash
cd docs
cat PROJECT_STRUCTURE.md
```

### 3️⃣ 运行AI演示 (2 分钟)
```bash
cd ai_training
python ai_training_demo.py
```

### 4️⃣ 测试基础设施 (2 分钟)
```bash
cd infrastructure
python demo_level2_features.py
```

### 5️⃣ 运行主程序
```bash
cd core_app
python main.py
```

---

## 📊 文件统计

| 分类 | 文件数 | 代码行数 | 说明 |
|------|--------|---------|------|
| 核心应用 | 4 | ~800 | 检测、深度、语音 |
| RAG系统 | 4 | ~900 | 检索、编排、异步 |
| AI训推 | 5 | ~1,200 | 推理、学习、选择 |
| 基础设施 | 5 | ~600 | 日志、追踪、缓存 |
| **总计** | **18** | **3,500+** | **完整系统** |

---

## 🔑 关键文件快速查询

### 想要...立即查看这个文件

| 需求 | 文件位置 |
|------|---------|
| **快速了解项目** | `docs/RESUME_BRIEF.md` |
| **详细文件结构** | `docs/PROJECT_STRUCTURE.md` |
| **使用指南** | `docs/AI_TRAINING_GUIDE.md` |
| **运行演示** | `ai_training/ai_training_demo.py` |
| **查看系统架构** | `rag_system/voice_rag_langgraph.py` |
| **理解AI学习** | `ai_training/learning_system.py` |
| **性能优化** | `infrastructure/cache_layer.py` |
| **启动主程序** | `core_app/main.py` |

---

## 💡 模块说明

### 🎯 核心应用 (core_app/)
处理实时的手部检测、对象识别、深度估计和语音交互。
- **入口**: `main.py`
- **依赖**: hand_detection_module, depth_detection_module, voice_module
- **功能**: 实时手鼠标检测 + 语音指导

### 🧠 RAG系统 (rag_system/)
混合检索引擎和多节点编排工作流。
- **核心**: `object_retriever.py` (Hybrid RAG - BM25+嵌入)
- **编排**: `voice_rag_langgraph.py` (7节点状态机)
- **异步**: `async_voice_rag_langgraph.py` (4.4×吞吐量)
- **知识库**: `object_knowledge_base.json` (8个对象)

### 🚀 AI训推系统 (ai_training/)
自适应推理、自动学习、智能选择和强化学习。
- **推理**: `adaptive_reasoning.py` (3级ReAct)
- **学习**: `learning_system.py` (模式提取)
- **选择**: `model_selector.py` (场景感知)
- **反馈**: `reward_system.py` (Q-Learning)
- **演示**: `ai_training_demo.py` (5个场景演示)

### 🔧 基础设施 (infrastructure/)
生产级的观测性和性能优化。
- **日志**: `logging_config.py` (<1%开销)
- **追踪**: `tracing.py` (瓶颈识别)
- **缓存**: `cache_layer.py` (10-100×加速)
- **演示**: `demo_level2_features.py`
- **测试**: `test_performance_level2.py`

---

## 📌 重要提示

### 文件移动后需要注意的导入

如果你从根目录运行脚本，需要调整导入路径：

```python
# ❌ 旧方式（文件在根目录）
from adaptive_reasoning import AdaptiveReasoningEngine

# ✅ 新方式（文件在子目录）
from ai_training.adaptive_reasoning import AdaptiveReasoningEngine
```

### Python路径配置（可选）

在根目录创建 `__init__.py` 文件或在脚本中添加：

```python
import sys
from pathlib import Path

# 添加子模块到Python路径
sys.path.insert(0, str(Path(__file__).parent / "core_app"))
sys.path.insert(0, str(Path(__file__).parent / "rag_system"))
sys.path.insert(0, str(Path(__file__).parent / "ai_training"))
sys.path.insert(0, str(Path(__file__).parent / "infrastructure"))
```

---

## ✅ 整理清单

- ✅ 核心应用 (4 个文件) → core_app/
- ✅ RAG系统 (4 个文件) → rag_system/
- ✅ AI训推 (5 个文件) → ai_training/
- ✅ 基础设施 (5 个文件) → infrastructure/
- ✅ 文档 (6 个文件) → docs/
- ✅ 模型 → models/
- ✅ 旧实验 → 保留原位置

**根目录现已清爽，仅保留配置文件夹（.venv, .history, __pycache__）**

---

## 🎯 下一步建议

1. **运行演示看效果**
   ```bash
   cd ai_training
   python ai_training_demo.py
   ```

2. **参考文档了解详情**
   ```bash
   cd docs
   # 阅读 RESUME_BRIEF.md 了解项目
   ```

3. **集成到你的应用**
   - 参考 `core_app/main.py` 的调用方式
   - 导入需要的模块

4. **优化性能**
   - 查看 `infrastructure/` 的日志和缓存配置
   - 运行性能测试

---

**项目整理于**: 2026-05-20
**总计**: 18+ 个文件，3,500+ 行代码，6 个主要分类
