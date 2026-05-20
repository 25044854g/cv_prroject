# 项目结构整理

## 📦 完整目录结构

```
computer_version_team_project/
│
├── 🎯 核心应用模块 (主系统)
│   ├── main.py                          # 主程序入口 (手鼠标检测 + 语音引导)
│   ├── hand_detection_module.py          # 手部+对象检测 (MediaPipe + YOLO-Pose)
│   ├── depth_detection_module.py         # 深度估计 (立体匹配)
│   └── voice_module.py                   # 语音识别 + LLM 语义映射 (Whisper + OpenRouter)
│
├── 🧠 RAG + LangChain 系统 (原始检索/编排)
│   ├── voice_rag_langgraph.py            # 7节点 LangGraph 状态机
│   ├── async_voice_rag_langgraph.py      # 异步包装器 (4.4× 吞吐)
│   ├── object_retriever.py               # Hybrid RAG 引擎 (BM25+嵌入)
│   └── object_knowledge_base.json        # 8个对象知识库
│
├── 🚀 AI 训练+推理系统 (新增高级功能)
│   ├── adaptive_reasoning.py             # ReAct 自适应推理 (3级链)
│   ├── learning_system.py                # 自动学习系统 (模式提取)
│   ├── model_selector.py                 # 智能模型选择 (场景感知)
│   └── reward_system.py                  # Q-Learning 强化学习 (多源奖励)
│
├── 🔧 生产级基础设施 (Level 2 - 可观测)
│   ├── logging_config.py                 # 结构化 JSON 日志 (8事件类型)
│   ├── tracing.py                        # 分布式追踪 (瓶颈检测)
│   └── cache_layer.py                    # 多层 LRU 缓存 (10-100×加速)
│
├── 📊 演示和测试
│   ├── ai_training_demo.py               # AI 系统完整演示 (5 场景)
│   ├── demo_level2_features.py           # Level 2 基础设施演示
│   └── test_performance_level2.py        # 性能测试
│
├── 📖 文档
│   ├── RESUME_BRIEF.md                   # 简洁简历版 (4方向陈述)
│   ├── AI_TRAINING_GUIDE.md              # AI 系统集成指南
│   ├── AI_TRAINING_SUMMARY.md            # AI 系统功能总结
│   ├── LEVEL2_INFRASTRUCTURE.md          # 基础设施详细文档
│   ├── LEVEL2_QUICKSTART.md              # 快速开始指南
│   └── README.md / readME.txt            # 原始项目文档
│
├── 📁 learning_data/                     # 学习数据持久化
│   └── learning_state.json               # 自动保存的学习进度
│
├── 📁 测试和实验文件夹
│   ├── basic_test_for_prepare/           # 初步 CV2 测试
│   │   ├── prepare for cv2 test1 read a picture.py
│   │   ├── prepare for cv2 test2 camera.py
│   │   └── prepare for cv2 test3 test_rectangle.py
│   │
│   ├── deeep_detection/                  # 深度学习测试
│   │   └── deep_test.py
│   │
│   └── space relation detection/         # 空间关系检测 (v1-v4 迭代)
│       ├── hand_object_combined.py       # v1 初版
│       ├── hand_object_combined_v2.py    # v2 改进
│       ├── hand_object_combined_v3.py    # v3 优化
│       ├── hand_object_combined_v4.py    # v4 最终版
│       ├── hand_object_combined_final.py # Final 版
│       ├── hand_object_hybrid.py         # 混合方案
│       ├── hand_object_mediapipe_final.py
│       ├── hand_object_mediapipe_final_yolom.py
│       ├── hand_object_yolo_pose.py
│       ├── mediapipe_final_yolom_no_person.py
│       ├── download_model.py
│       ├── hand_landmarker.task
│       ├── yolov8m.pt
│       ├── yolov8n.pt
│       └── readme.txt
│
│   └── the detection prapare/            # 检测准备模块
│       ├── hand_detection.py
│       ├── object_detection_coordinates.py
│       └── object_detection_yolo8.py
│
├── 🤖 模型文件
│   ├── yolov8m.pt                        # YOLO v8 中等模型
│   ├── yolov8n-pose.pt                   # YOLO v8 姿态模型
│   └── hand_landmarker.task              # MediaPipe 手部模型
│
├── 🔙 虚拟环境和缓存
│   ├── .venv/                            # Python 虚拟环境
│   ├── __pycache__/                      # Python 缓存
│   ├── .history/                         # VS Code 历史
│
└── 📋 依赖文件
    ├── requirements.txt                  # Python 依赖列表
    └── .gitignore / git config
```

---

## 📊 模块分类统计

### 生产代码 (16+ 模块，3,500+ 行)

| 类别 | 模块数 | 代码行数 | 功能 |
|------|--------|---------|------|
| 核心应用 | 4 | ~800 | 主程序、检测、深度、语音 |
| RAG系统 | 4 | ~900 | 检索、编排、异步、知识库 |
| AI训推 | 4 | ~1,200 | 推理、学习、选择、奖励 |
| 基础设施 | 3 | ~600 | 日志、追踪、缓存 |
| **总计** | **16+** | **3,500+** | **完整系统** |

### 文档 (6 份，20+ KB)
- RESUME_BRIEF.md (1.2 KB) - **简洁简历**
- AI_TRAINING_GUIDE.md (11.9 KB) - 集成指南
- AI_TRAINING_SUMMARY.md (10.9 KB) - 功能总结
- LEVEL2_INFRASTRUCTURE.md - 详细文档
- LEVEL2_QUICKSTART.md - 快速开始
- README.md - 原始文档

### 演示和测试 (3 份)
- ai_training_demo.py - 5 个完整场景
- demo_level2_features.py - 基础设施演示
- test_performance_level2.py - 性能测试

### 实验文件夹 (3 个，多个版本)
- basic_test_for_prepare/ - 初步测试
- deeep_detection/ - 深度学习实验
- space relation detection/ - 迭代版本 (v1-v4)
- the detection prapare/ - 检测模块

---

## 🎯 快速导航

### 如果你想...

**学习整个系统**
→ 从 `RESUME_BRIEF.md` 开始 (3 分钟概览)

**理解 AI 系统**
→ 读 `AI_TRAINING_GUIDE.md` + 运行 `ai_training_demo.py`

**集成到实际应用**
→ 参考 `main.py` + `voice_rag_langgraph.py` 的调用方式

**性能优化**
→ 查看 `cache_layer.py` + `tracing.py` + `test_performance_level2.py`

**学习如何工作**
→ 运行 `demo_level2_features.py` 看完整演示

**准备面试**
→ 用 `RESUME_BRIEF.md` + 理解 4 大模块的原理

---

## 💡 核心文件说明

### 必读文件
1. **RESUME_BRIEF.md** - 项目概览 (1.2 KB，包含所有关键数据)
2. **AI_TRAINING_GUIDE.md** - 使用指南 (含代码示例)
3. **main.py** - 系统入口 (展示如何调用各模块)

### 关键模块
1. **voice_rag_langgraph.py** - 流程编排核心
2. **object_retriever.py** - RAG 检索核心
3. **adaptive_reasoning.py** - AI 推理核心
4. **learning_system.py** - 自动学习核心

### 演示文件
1. **ai_training_demo.py** - 直接运行看效果
2. **demo_level2_features.py** - 基础设施演示

---

## 📌 版本历史

| 阶段 | 模块 | 特点 |
|------|------|------|
| **基础** | 检测、深度、语音 | 基本功能 |
| **RAG** | 检索、编排、异步 | 85% 准确率 |
| **Level 2** | 日志、追踪、缓存 | 生产级基础设施 |
| **AI** | 推理、学习、选择、奖励 | 自适应自学习 |

