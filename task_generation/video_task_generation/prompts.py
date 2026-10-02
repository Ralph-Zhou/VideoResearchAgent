"""Prompt templates for the Video DeepResearch task generation pipeline."""

# ── Stage 1: seed entity generation ──────────────────────────────────
#
# Generate seed entities that are plausible video subjects: sporting events,
# concerts, documentaries, vlogs, tutorials, product reviews, and so on. The
# goal is concrete proper nouns that work as YouTube queries for real videos.

SEED_ENTITY_CATEGORIES = {
    "体育赛事": [
        "田径锦标赛", "足球顶级联赛经典比赛", "篮球季后赛经典对决",
        "网球大满贯名场面", "F1 赛段经典圈速", "奥运特定项目决赛",
        "电竞世界赛关键局", "格斗运动名局", "冬奥赛事",
    ],
    "演唱会与现场音乐": [
        "乐队巡演 live", "独立音乐节现场", "古典乐团音乐会",
        "爵士音乐节", "说唱 cypher 现场", "民谣专场",
        "音乐剧 live recording", "户外大型演唱会",
    ],
    "纪录片与专题报道": [
        "自然生态纪录片", "历史事件重访", "城市人文纪录",
        "手艺人纪录", "极限环境探险", "科学实验纪录",
        "濒危物种观察", "考古发掘纪实",
    ],
    "vlog 与旅行": [
        "小众目的地旅行", "徒步穿越 vlog", "城市漫游",
        "美食探店", "异国文化体验", "背包客长途记录",
        "露营过夜", "小镇节庆旅拍",
    ],
    "产品评测与拆解": [
        "相机镜头实拍评测", "耳机与音响听感", "无人机飞行测试",
        "电动工具耐用性", "智能家居设备", "机械键盘实测",
        "户外装备拆解", "老式数码产品翻新",
    ],
    "教学与操作演示": [
        "实验室科学演示", "厨房烹饪教学", "手工艺教程",
        "编程实操录屏", "乐器教学", "健身训练示范",
        "急救与户外生存", "摄影 / 剪辑教程",
    ],
    "动物与宠物": [
        "野生动物观察", "极地动物跟拍", "濒危物种救援",
        "宠物训练", "水族生态缸", "昆虫微距",
    ],
    "机械与工程演示": [
        "大型工程机械作业", "古董机械修复", "铁路运行录影",
        "船舶下水", "老式发动机启动", "钟表机芯拆解",
    ],
    "文化与表演": [
        "传统戏曲演出", "民俗节庆仪式", "街头表演艺术",
        "杂技与马戏", "舞蹈编排录像", "非遗技艺演示",
    ],
    "实况与直播精选": [
        "体育实况解说", "游戏实况经典场面", "新闻现场连线",
        "天气奇观直播", "灾害救援记录", "天文现象直播",
    ],
}

REGION_MODIFIERS = [
    "中国大陆", "中国香港", "中国台湾", "日本", "韩国", "东南亚",
    "印度", "中东", "欧洲", "北美", "拉美", "非洲", "大洋洲",
]

ERA_MODIFIERS = [
    "近五年", "2010-2020", "2000-2010", "1990年代", "1980年代", "再更早时期",
]

# Language dimension used when generating seeds. The caller samples zh / en at a
# target ratio so downstream task synthesis produces both Chinese and English
# questions.
SEED_LANGUAGES = ("zh", "en")


SEED_GENERATION_PROMPT_ZH = """你在为一个 **Video Deep-Research 任务合成系统** 生成 seed entity。
下游会用这些 entity 作为 YouTube 搜索 query，必须能搜到真实存在的视频。

## 本次聚焦
- 类别: {category}
- 子类别: {subcategory}
- 地区/背景: {region}
- 时代: {era}

## 要求
1. 输出 {batch_size} 个**具体、真实存在**的 seed entity（人名、比赛/事件名、作品名、产品型号、地点事件等），而不是笼统类别。
2. 每个 entity 必须同时满足：
   - 有**较大概率**在 YouTube 上能搜到专门的视频（巡演、纪录片、评测、讲解、实况等）。
   - 不是全球头部热词（比如"梅西""iPhone"这种），而是**偏中长尾**的具体实例。
3. **语言约束（强制）**：本批次必须全部输出**中文**名称的 entity（简体或繁体皆可）。
   不要输出英文 entity。若该实体本身没有中文名，跳过不输出。
4. 每行只输出一个 entity，不带序号、引号、解释、多余标点。
{exclusion_clause}
## 输出
仅一列中文 entity，每行一个。
"""


SEED_GENERATION_PROMPT_EN = """You are seeding a **Video Deep-Research task synthesis system**.
Downstream, each seed becomes a YouTube query that must retrieve a real video.

## Focus
- Category: {category}
- Subcategory: {subcategory}
- Region/background: {region}
- Era: {era}

## Requirements
1. Output {batch_size} **concrete, real-world** seed entities (person names, event / match
   names, work titles, product models, specific places/incidents), NOT generic categories.
2. Each entity must satisfy:
   - High likelihood of having **dedicated YouTube videos** (tour recordings, documentaries,
     reviews, explainers, live feeds, etc.).
   - NOT a global top-tier household name (avoid "Messi", "iPhone"), prefer **mid / long tail**
     specific instances.
3. **Language constraint (hard)**: this batch must ONLY contain **English** entity names
   (Latin-alphabet). Do NOT output Chinese entities. Skip any entity that does not have a
   widely-used English name.
4. One entity per line, no numbering, quotes, commentary, or extra punctuation.
{exclusion_clause}
## Output
Just the list, one English entity per line.
"""


# Backwards-compat alias — the zh version is the original.
SEED_GENERATION_PROMPT = SEED_GENERATION_PROMPT_ZH


def seed_generation_prompt_for(language: str) -> str:
    """Return the correct seed-generation template for a language code."""
    lang = (language or "zh").lower()
    if lang.startswith("en"):
        return SEED_GENERATION_PROMPT_EN
    return SEED_GENERATION_PROMPT_ZH


# ── Stage 2a: per-frame caption ──────────────────────────────────────

FRAME_CAPTION_SYSTEM = (
    "You are a careful frame annotator. Given a single frame sampled from a video, "
    "describe what is visually happening, including objects, people, on-screen text, "
    "setting, and any notable actions. Avoid speculation; describe only what you see."
)

FRAME_CAPTION_USER = (
    "Frame timestamp: t={timestamp:.2f}s.\n"
    "Video title: {title}\n"
    "Associated entity being researched: {entity}\n\n"
    "请用中文输出这一帧的精炼视觉描述，限制在 {max_chars} 字以内。"
    "要求：\n"
    "1. 列出画面中**具体可见**的物体/人物/场景；\n"
    "2. 如有**可辨认的屏幕文字 / 字幕 / logo / 时钟数字**，逐字摘录；\n"
    "3. 不要主观推测未在画面中出现的信息；\n"
    "4. 不要复述本提示词。"
)


# ── Stage 2c: entity enrichment (properties + relations) ─────────────
#
# Run one lightweight text search per real hop entity in the graph and collect:
#   - properties: 5-8 concrete, verifiable factual attributes;
#   - relations:  3-6 mappings of {related entity name: relationship description}.
# The results go back into VideoEntity.properties / .relations. When Stage 3
# generates a task it can pick 2-3 properties as indirect descriptive
# constraints that narrow the target entity down to a unique referent.
#
# Key constraints:
#   * properties must be phrased as factual statements, avoiding subjective
#     wording such as "well-known" or "famous";
#   * visual detail at the video-frame level (frame captions) must not leak in,
#     so these never conflict with frame_evidence;
#   * never emit a complete "giveaway" phrase that would surface the answer in
#     a single search.

ENTITY_ENRICHMENT_SYSTEM_ZH = (
    "你是严谨的百科编辑。你会收到一个实体名称和若干网页片段，"
    "请从中抽取该实体的**事实性背景属性**与**关联关系**。要求：\n"
    "1) properties 必须是**具体、可查证**的陈述句（例如所属国家/行政区、成立/发行年份、"
    "   核心参数、代表作品、官方归属机构、地理坐标/面积/人口等）；\n"
    "2) **禁止**使用“知名”“著名”“备受好评”“颇具影响力”这类主观评价词；\n"
    "3) **禁止**泄露视频画面、帧截图、banner、弹幕、视频 UI 里才能看到的视觉信息；\n"
    "4) relations 的每个 key 是一个其他实体的专名，value 是一句话描述和目标实体的关系；\n"
    "5) 若部分信息搜不到或不确定，宁可**少输出**也不要编造。"
)

ENTITY_ENRICHMENT_SYSTEM_EN = (
    "You are a rigorous encyclopedia editor. You will receive an entity name and "
    "several web snippets. Extract **factual background properties** and **relations**.\n"
    "Requirements:\n"
    "1) Each property must be a concrete, verifiable declarative statement (country / "
    "   administrative division, founding / release year, core specs, representative works, "
    "   official governing body, geographic coordinates / area / population, etc.);\n"
    "2) NEVER use subjective adjectives like 'famous', 'renowned', 'well-acclaimed';\n"
    "3) NEVER reveal visual details that only appear inside a video frame / thumbnail / "
    "   banner / UI — stick to textually verifiable facts;\n"
    "4) Each relation key is another entity's proper name; the value is a one-line "
    "   description of its relationship to the target entity;\n"
    "5) When in doubt, output FEWER items rather than fabricate."
)


ENTITY_ENRICHMENT_USER = """## 目标实体 / Target entity
{entity_name}

## 视频中对该实体的观察摘要（供参考，**不要**把其中的画面细节写进 properties）
{video_summary}

## Web snippets（来自文本搜索，可能含噪）
{snippets}

## 任务
1. 综合 snippets 给出 **5-8 条 properties**（每条一个事实陈述，句末无句号或逗号均可）；
2. 给出 **3-6 条 relations**：key=其他实体专名，value=和目标实体的关系描述；
3. 输出**严格 JSON**，UTF-8，键名固定为 `properties` / `relations`。

## 输出格式（只输出这一份 JSON，不要多余文字）
```json
{{
  "properties": [
    "属性 1 ...",
    "属性 2 ..."
  ],
  "relations": {{
    "相关实体 A": "和目标实体的关系 ...",
    "相关实体 B": "关系 ..."
  }}
}}
```
"""


# ── Stage 2b: video-level synthesis + next-entity pick ───────────────

VIDEO_SYNTHESIS_PROMPT = """你将收到一段从视频均匀抽取的逐帧描述（共 {n_frames} 帧，按时间顺序）。

## 视频基本信息
- 标题: {title}
- 频道: {channel}
- 时长: {duration} 秒
- 所研究的 entity: {entity}

## 逐帧描述
{frame_captions}

## 任务
1. 先综合所有帧，给出一段 **6-10 句** 中文**视频综合描述**，覆盖：
   - 视频主题和总体走向
   - 关键画面中出现的专有名词（人名、地名、比赛、设备、作品等）
   - 任何辨识出的屏幕字幕/数字/时间戳/得分等"锚定事实"
2. 从视频综合描述中**挑选一个新的文本 entity**（专有名词），作为 entity graph 的下一跳：
   - 它必须是一个真实存在、可在 YouTube 重新搜到视频的具体名词；
   - 它与当前 entity "{entity}" 有自然关联（比如同场比赛的另一方、视频中出现的另一个装备/人物/地点/作品/事件）；
   - **不要**选当前 entity 自己，也不要选过于泛化的普通名词（如"足球""城市"）。

## 输出格式（严格遵守）
<summary>
[视频综合描述]
</summary>
<next_entity>
[下一跳的具体 entity 名称，仅一个词或一个专有短语]
</next_entity>
<reason>
[简短说明为什么选它，它与当前 entity 的关系]
</reason>
"""


# ── Stage 3a: task generation from VideoEntityGraph ──────────────────

_TASK_LANGUAGE_INSTRUCTION = {
    "zh": (
        "**语言约束（强制）**：本题的 seed entity 为**中文**。请**必须使用中文**撰写 "
        "<thinking>、<question>、<answer>、<frame_evidence> 四个字段的全部内容。"
        "answer 若为英文专有名词（品牌、产品型号、人名拉丁转写等）可保留原文；"
        "其余叙述一律中文。禁止整段英文表述。"
    ),
    "en": (
        "**Language constraint (hard)**: the seed entity for this task is in **English**. "
        "You MUST write the entirety of <thinking>, <question>, <answer>, and "
        "<frame_evidence> in **English**. If the answer is a non-English proper noun "
        "(Chinese name, kanji title, etc.), you may keep the original script, but the "
        "surrounding narration must remain English. Do NOT switch to Chinese narration."
    ),
}


def task_generation_language_instruction(language: str) -> str:
    lang = (language or "zh").lower()
    if lang.startswith("en"):
        return _TASK_LANGUAGE_INSTRUCTION["en"]
    return _TASK_LANGUAGE_INSTRUCTION["zh"]


TASK_GENERATION_PROMPT = """你是一个 **Video DeepResearch 任务设计师**。
下面给你一条 entity graph trajectory：沿着这条 trajectory，每一跳都链接到一个具体的 YouTube 视频，
并配有**帧级视觉证据**（逐帧 caption）+ **视频综合描述**。

## Entity Graph Trajectory
{graph_text}

## 风格范例（**涵盖不同问点类型**，你要从中选一个合适的类型来构造新 task）
下列示例展示了 7 种**不同的视觉问点**（文字只是其中一种，不要过度偏好）：

- 示例 1【画面文字】（中文）：
  "有一个发布过经典大象牙膏碘化钾催化亲子 STEAM 实验的频道，该频道后来和专业教育机构合作
  推出过一日职业体验类视频，在这个体验视频中，参与体验的小男孩进行播报工作时所穿短袖上，
  印着的文字是什么？"
- 示例 2【物体计数 / 数量，类别 + 位置限定词】（英文）：
  "There is a channel known for covering historical aviation airshows. In its
  commemoration video for a famous 1947 transatlantic flight, during the opening
  overhead shot of the apron, how many **propeller-driven aircraft** are **parked on
  the right side of the runway** (counting any aircraft whose full body fits within
  the frame, excluding those cropped by the frame edge)?"
  ↑ 注意：限定词用"螺旋桨飞机"（客观类别）+ "跑道右侧"（客观位置）；**不用**
  "真正的老式战机""清晰可辨的飞机"这种语义模糊词；**并且**显式写明边缘截断怎么处理。
- 示例 2b【物体计数，颜色 + 动作限定词】（中文）：
  "有一档游戏解说频道发布过某款经典体育竞技游戏的开场 CG 解析视频，在 CG 里全体
  角色在体育场入场列队亮相的那个全景镜头中，**穿红色上衣且正在挥手的角色**一共有几位
  （被前景人物遮挡过半的不计入）？"
  ↑ 注意："红色上衣"+ "正在挥手"是两个 VLM 客观可枚举的限定词；**没有**"完整露出
  身形""算得上主角"这种对 VLM 模糊的词；遮挡阈值也给死了。
- 示例 3【颜色 / 外观属性】（中文）：
  "有一档介绍非遗竹编技艺的纪录短片，在介绍某位传承人日常工作场景的那段里，传承人把成品
  竹篮放到工作台中央展示时，这只竹篮的主色调是什么？"
- 示例 4【动作 / 手势 / 互动】（英文）：
  "There is a well-known cooking demo channel that once released a Sichuan-style
  braised-pork tutorial. In the tasting section of that episode, when the host takes
  the first bite, with which hand does she hold the chopsticks?"
- 示例 5【空间 / 相对位置】（中文）：
  "有一支介绍某内蒙古草原那达慕大会的官方宣传片，在开幕式方阵入场的俯拍镜头中，最靠近镜
  头的那面彩旗位于画面的哪个方位（左/中/右）？"
- 示例 6【顺序 / 时间点】（英文）：
  "There is a mini-documentary introducing a classic American college marching band's
  halftime show. In the show's opening formation, which letter of the school's name do
  the performers spell out FIRST before morphing into the next shape?"
- 示例 7【UI / 数字读数】（中文）：
  "有一档汽车自媒体发布过某款纯电跑车的赛道测试视频，在车内仪表回放画面中，车辆冲过终点
  那一瞬间，中控屏幕上显示的最高实时时速是多少公里/小时？"

这些示例共同遵循的风格：
- **一段通顺叙事**，不用条目；
- **刻意省略**"海外知名视频平台""YouTube""a major global video-sharing platform"这
  类套话——因为 task 本身已经明确让人找视频，再强调平台显得拗口。你也必须**省掉**这种
  开场白（见 B.5）。
- **措辞不要千篇一律**：不要每个 task 都以"清晰可辨认的…是什么"收尾；问法应与**视觉问点
  的类型**相匹配（数量→"多少个/几张"；颜色→"是什么颜色/主色是"；动作→"做了什么动作/用
  的是哪只手"；顺序→"先出现的是哪一个"；等）。
- **绝对不能暴露 entity graph 的内部构造逻辑**——下面 B.7 有详细规定。

## 设计任务（硬性要求）

### A. 视频必需性（与旧版一致，仍然硬约束）
1. **必须依赖视频帧**：核心关键的一步推理**只能**通过观察视频画面（屏幕文字、物体、数字、比分、
   LOGO、动作、布景等）才能得到。纯文本搜索引擎通过标题/描述/字幕文本**无法**解出。
2. **多跳推理**：
   - 如果 graph depth ≥ 2：task 必须跨越至少 2 个 entity，显式包含"实体-关系"跳转；
   - 如果 graph depth = 1：围绕视频"先用文本线索定位到该视频 → 再看某一帧 → 推到答案"构造。
3. **不要直接点明视频/频道**：question 不能直接写出视频 URL、video id、视频标题原文、频道
   名字 / 账号等硬线索；必须用"某频道""某期节目""有一支…主题的视频"等**间接描述**让 agent
   通过关联推理去找到视频。

### B. 文风（必须接近上面的示例）
1. **连贯叙事**：用一段通顺中文（或英文）把"起点实体 → 关联跳转 → 目标视频 → 视频内的具体
   瞬间 → 所问事实"一气讲清楚；避免条目化/公式化表达。
2. **自然的关键动词**：偏好"发布过…""曾推出过…""该频道与…合作…""在这期视频中…"等自然
   语言，**但不要**把每个 task 都写成"清晰可辨认的…是什么"——这是一种常见提问模板，过度
   使用会让数据集同质化。请根据 B.2b 的视觉问点类型选择与之**语义相配**的问法。
2b. **视觉问点类型（务必轮换）**：你的问题结尾问的是**视觉事实 X**，X **必须**属于下面某一
   类，不要总选"画面文字"：
   (i) **画面文字**：屏幕/服装/标识/字幕/铭牌上的可辨认文字；
   (ii) **数量 / 计数**：画面中某类对象的个数（人数、车数、气球数、灯珠数等）。
        **限定词必须是 VLM 能够按像素/区域直接判断的客观视觉属性**，而**不是**语义/认知层
        面才能区分的概念。见下方 C.6 的硬约束。
   (iii) **颜色 / 外观属性**：主色、条纹方向、材质观感（确保**单一客观**，不问主观审美）；
   (iv) **动作 / 姿势 / 手势**：人物做了什么动作、用哪只手、朝哪个方向等；
   (v) **空间 / 相对位置**：某对象位于画面左/中/右或前景/背景、相对另一对象的方位；
   (vi) **顺序 / 时间点**：一组画面里先/后出现的那一个；镜头由 A→B→C 的切换顺序；
   (vii) **UI / 数字读数**：仪表盘数字、比分、时间戳、评论/播放/粉丝等界面数字；
   (viii) **形状 / 图案 / 符号**：几何形状、徽章图案、标志物轮廓等可命名的视觉结构；
   ——同一批次中**同类型不要连续出现**；如果 entity graph 的画面天然只适合"画面文字"，
   仍可选 (i)，但**问法措辞**要换（"印着的文字是什么""上方的标语是""招牌正中写的是"等），
   避免统一写成"清晰可辨认的…文字是什么"。
3. **锚点选择**：保留足够让人类/agent 能联想到正确视频的描述性锚点（类别、题材、合作方角色、
   场景描述），**但不要**出现让搜索引擎一击命中答案的"致命线索"（如专有账号、罕见完整片名、
   上传日期精确到日）。必要时做**最小模糊**即可（例如"2022 年 3 月"→"近几年"），**不要为了
   模糊而把自然语言改得拗口**。
4. **精准描述问点**：对要问的视觉细节要具体定位到**画面中某个独一无二的对象**：如"参与体验的
   那位小男孩**播报工作时**所穿的**短袖正面**""测试框架固定后的**那块防弹插板正面**""商场
   入口右侧的电子屏**第一行字**"等——让读者一看就知道要盯哪里。
5. **不要提"视频平台"本身**。以下这类套话**必须省掉**，不要出现在 question 里：
   - 中文：**禁止**"海外知名视频平台上""YouTube 上""某视频网站上""互联网视频平台上"等；
   - 英文：**禁止**"On a major global video-sharing platform…""On YouTube…""On a
     popular video website…"等。
   直接以"有一个…频道""有一支围绕…的视频""某档节目"等自然起笔即可。任务上下文已经默认
   agent 需要去找视频，无需画蛇添足。
6. **避开"水印/频道角标/平台标识"作为答案**：这类元素容易被误读为"底部文字"但并非视频内容
   本体。如果你选了 (i) 画面文字类问点，**必须**在 question 里明确说明要问的是"画面主体内
   的文字"（如"字幕内容""讲台上标语""展板正中文字""人物服装上的字"），而**不是**通用水
   印、频道 logo、平台右下角图标等。
7. **严禁暴露 entity graph 的 hopping 逻辑** ★★★（违反即废弃）。
   以下这些表达方式泄露了内部的 graph 构造过程，必须**彻底杜绝**：
   - **中文禁用**："原本研究的是…""顺着自然关联跳转…""沿着…的自然关联""跳转到…""由此
     联想到…""从…过渡到…""顺着…消费场景的关联""通过…的跳转关系""实体之间的连接"等一切
     暴露"多跳搜索策略"或"graph hop"过程的表达；
   - **英文禁用**："Following a natural association…""Jumping from… to…""Transitioning
     from… we find…""Hopping through…""The entity connects to…"等；
   - **正确做法**：直接描述**目标视频的内容和特征**，让读者/agent 自己去搜。把多跳关系
     转化为**背景事实的叙述**。对比：
     ✗ "广东沙溪古镇是国内南粤柔道运动发展的核心区域，顺着运动爱好者消费场景的自然关联，
       能找到一款知名高端腕表品牌…"
     ✓ "有一支高端腕表品牌中国限定款的开箱视频，该品牌曾赞助过国际柔道赛事。在开箱展示
       腕表本体的镜头中，表带的主色调是什么？"
   - **检验标准**：如果你把 question 单独给一个不知道 entity graph 的人看，他不应该能从
     措辞中推断出"这题是从 A 跳到 B 再跳到 C"的三跳结构——他只应该看到一个需要搜索和
     看视频才能回答的自然问题。

### C-pre. 用 properties 把目标 entity 间接收敛到唯一 ★重要
每个 hop 可能带有一段 "Background properties" 和 "Named relations"（来自文本搜索
汇总）。这些是**事实性、可查证、不涉及画面细节**的背景属性，你要**充分利用**：
1. 不要在 question 里**直接写出**目标 entity 的专有名（如"Chur""Starks Shooting Complex"），
   而是用 **2-3 条 properties 组合**作为**间接描述性约束**来指向它。例如用：
   "有一座位于瑞士东部某州、州首府之一且海拔接近 600 米的小城"
   替代直接说 "Chur"；
2. 挑选的 properties 应满足：
   (a) **组合起来足以唯一锁定**该 entity（否则目标会漂移）；
   (b) **单独任一条**又不会被直接拿去 Google 就出答案（例如只说"瑞士的小城"太泛，而
       "瑞士某州首府、海拔 593 米、面积约 28 平方千米"这种堆叠三条就几乎一击必中，
       需要**适度模糊**，比如把 593 m 写成"海拔接近 600 米"、28 km² 写成"不到 30
       平方千米"）；
   (c) **不要**照抄 properties 原文——换成自然语言复述；
3. relations 可用来指明"这一跳的来源实体"（"在记录 X 场景的某部作品中出现的地点"），
   同样要做自然化复述，不要直接点名；
4. **禁止把 properties 当答案用**：properties 是为了"定位 entity"，最终 answer 仍必须
   来自**视频帧级视觉证据**（见 A 与 C）；
5. 输出时把你实际使用的 properties 在 <used_properties> 中逐条列出（原文或你改写后的
   表达都行），以便下游自动化诊断。

### C. 唯一性 / 排他性（硬约束，务必满足）★重要
这是新增的硬约束，违反即视为无效 task：

1. **答案必须唯一且正向**：答案应是"在指定画面中**唯一可辨认**"的具体事物（一个名字、一个
   数字、一串文字、一个品牌、一个编号等）。
2. **严禁"负向穷举/缺失型"问法**。反例（**不允许**）：
   - "…**没有出现**哪个数字/字母/颜色？"
   - "以下选项中**缺失的**是哪一个？"
   - "**不包含**下列哪项？"
   这类问法的答案集合无穷或存在多解（比如画面上编号是 1, 2, 4, 5, 6，问"缺了哪个"，3、7、
   8、9…都算对）。正确做法是**正向提问**：改问"画面中**可辨认的最大编号**是多少""**按从左
   到右顺序出现的第一个编号**是多少"等。
3. **限定唯一指代**。当要问的对象可能在视频里出现多次（如"某品牌""某个人"），必须用
   限制语把它**收窄到单一实例**。例如：
   - 不说"视频里出现的品牌名是什么"（可能有多个）；
   - 而说"在主持人近景镜头中，桌面左上角瓶身上的品牌名是什么"。
4. **回避主观/可争议的描述词**。不要让答案依赖"最漂亮""最突出""最显眼"这类主观形容；应
   使用**客观、可机械验证**的定位（颜色、位置、顺序、计数、文字内容、编号等）。
5. **帧证据必须充分支撑**：在 <frame_evidence> 里须说明"哪个 entity / 哪个时间戳的画面
   直接给出了这个答案"，且这个画面里**只**能推出一个答案。如果你发现帧证据支持多个合理答案，
   换一个提问点。

6. **计数类问点的限定词必须是 VLM 可直接枚举的视觉属性** ★重要（专治 (ii) 数量/计数）。
   下游的 video agent 依赖 VLM 读帧，VLM **强于**按颜色/形状/位置/动作/服饰/物体类别
   划分计数对象，**弱于**按模糊的语义限定（"完整露出"/"清晰可辨认"/"算得上主角的"/
   "真正意义上的"）推断边界。因此，凡是 (ii) 数量/计数 类问点：

   (a) **只允许**使用以下这类**客观可视觉枚举**的限定词：
       - **颜色/服饰**：红衣的人数、穿黑色西装的人数、戴口罩的人数；
       - **动作/姿势**：举手的人数、坐下的人数、指向镜头的人数、正在挥舞旗帜的人数；
       - **朝向/面部**：正脸对镜头的人数、低头的人数；
       - **位置/区域**：主席台上的人数、讲台前方的椅子数、舞台左半边的演员数；
       - **类别**：直升机数、红色轿车数、圆形标牌数、挂在墙上的画框数；
       - 以上可组合（"主席台上举手的人数"）。

   (b) **严禁**以下**对 VLM 语义模糊**的限定词：
       - "**完整露出身形**的 / 完整出镜的 / 完整呈现的"——"完整"的阈值不确定；
       - "**清晰可辨认**的 / 清晰可见的 / 能看清的"——"清晰"的阈值主观；
       - "**主要**的 / **真正**的 / **算得上** X 的 / **有代表性**的"——涉及人类理解性判断；
       - "**独立的**"（独立于谁？需要语境）、"**不同的**（贴纸/颜色）"（如果多张贴纸
         上数字相同算不算不同？——改成"带数字标号的贴纸张数"即可）；
       - "**可数的** / **可识别的**"——重复 (b) 的模糊性。

   (c) **边界要给死，不要给软**。如果计数对象在画面里存在"部分被遮挡/边缘截断"的个体，
       必须在 question 里**显式规定**如何处理（"**包括**部分被遮挡的"或"**不计入**被
       遮挡一半以上的"），不要留给 VLM 自己猜。

   (d) **答案稳健性自检**：如果你发现把限定词替换成另一种合理解释后答案会变动（比如
       "完整露出"算 5，"至少露头"算 8），说明限定词不够客观，必须换一个。

   (e) **数目不宜过多**：为了让 VLM 在单帧下得出稳定计数，目标答案**尽量在 2-10 之间**；
       超过 10 就容易 ±1 / ±2 抖动，除非限定词是"行数/列数"这类天然稳健的结构量。

### D. 答案形式
- 一个**简短、确定、可验证**的实体或事实（名字 / 数字 / 日期 / 完整词组）；
- 不要"多选一"、不要开放回答；
- 如果最自然的答案是一个词组（如 "Human heart anatomical model"），就保持短语形式；
- **不得**给出带"或"、"可能"、"大约"等含糊标记的答案。

### E. 语言
- {language_instruction}

## 思考步骤（强烈建议在 <thinking> 中写出）
- Step A: **先决定本次问点类型** —— 从 B.2b 的 8 种视觉问点类型 (i)-(viii) 中**挑一个与当
  前 entity graph 的画面最契合**的类型；**如果画面同时支持多种类型，优先选非 (i) 类**以提
  升数据集多样性。
- Step B: 从 entity graph 中挑一个"**只有观察视频画面才能得到、且在对应画面里独一无二**"的
  视觉事实 X（其类别由 Step A 决定）。在 <thinking> 里**自我审查**四点：
  (a) "如果我把 X 换成别的值，答案是不是也可以成立？" → 有歧义就换；
  (b) 如果选了 (i) 画面文字类，**确认问的不是水印/频道 logo/平台角标**；
  (c) 确认问法措辞没有落入"清晰可辨认的…是什么"这种同质化模板；
  (d) 如果选了 (ii) 数量/计数类，**逐条过 C.6 的 (a)-(e)**：限定词是不是
      颜色/动作/位置/类别这类 VLM 客观可枚举的？有没有用到"完整露出/清晰可辨认/
      主要的/真正的/独立的/不同的"这种语义模糊词？遮挡边界有没有显式规定？目标
      答案是不是在 2-10 之间？任一条不满足就**换限定词或换问点**。
- Step C: 从 seed entity 出发，设计"文本线索 → 锁定该视频 → 观察 X → 得出答案"的多跳叙事，
  风格贴近前面的 7 条示例。
- Step D: 给出确定答案。

## 输出格式（严格）
<thinking>
[1) 本次选的**问点类型**是 (i)-(viii) 中的哪一个，为什么这是与画面最契合的类型；
 2) 选了哪一帧 / 哪段视觉事实 X 作为关键；
 3) 自我审查：为什么这个提问在该画面下答案唯一；如选 (i) 文字类还需说明**不是水印/角标**。]
</thinking>
<question>
[一整段自然语言的问题，贴近示例 A / B 的风格]
</question>
<answer>
[简短、唯一、可验证的最终答案]
</answer>
<frame_evidence>
[1-3 句话指出：哪个 entity 的哪一帧（时间戳或序号）、画面里具体什么位置/对象承载了答案，
 以及它为什么**只**支持这一个答案。]
</frame_evidence>
<used_properties>
[逐行列出你在 question 里实际用到的 properties 条目（原文或你的复述皆可）；若没用
 任何 properties，留空行即可。]
</used_properties>
"""


# ── Stage 3b: text-only search agent for verification ────────────────

TEXT_SEARCH_AGENT_SYSTEM = """You are a text-only research agent that tries to answer a question using web search ONLY.

You have two tools:
1. <search>query</search>   — run a web search, returns a list of snippets+URLs
2. <visit url="...">goal</visit> — open a web page and extract information relevant to a goal

Rules:
- You CANNOT watch or analyse videos. You only have access to text on the web.
- You may issue multiple search/visit calls in sequence, each on its own turn.
- When you are confident of the final answer, output:
  <final_answer>
  [your best answer, or "UNANSWERABLE_WITHOUT_VIDEO" if you cannot find it via text alone]
  </final_answer>
- If after a genuine attempt you still can't recover the answer from text-only sources,
  output UNANSWERABLE_WITHOUT_VIDEO — do NOT guess.

Respond each turn with EITHER:
  <search>query</search>
  OR
  <visit url="...">goal</visit>
  OR
  <final_answer>...</final_answer>
Do not output anything else outside these tags.
"""

TEXT_SEARCH_AGENT_USER = """Question to answer using **text-only web research**:

{question}

Investigate step by step. Start by issuing a search query.
"""


# ── Stage 3c: obfuscation refinement ─────────────────────────────────

_OBFUSCATION_LANGUAGE_INSTRUCTION = {
    "zh": (
        "**语言约束（强制）**：原 question 为中文，refined_question 必须继续使用中文，"
        "不得整段切换为英文。"
    ),
    "en": (
        "**Language constraint (hard)**: the original question is in English. "
        "The refined_question MUST remain in English — do NOT switch the narration to Chinese."
    ),
}


def obfuscation_language_instruction(language: str) -> str:
    lang = (language or "zh").lower()
    if lang.startswith("en"):
        return _OBFUSCATION_LANGUAGE_INSTRUCTION["en"]
    return _OBFUSCATION_LANGUAGE_INSTRUCTION["zh"]


OBFUSCATION_PROMPT = """你是一个 task 文本润色器，而不是"强制模糊器"。下面的 question 已经通过
"纯文本无法解答 + 必须依赖视频帧"的双重验证，**总体可答性已经合格**。你的目标是：

1. 仅在**必要**时做**最小模糊**：把会让搜索引擎"一击命中答案"的硬线索（专有账号名、视频
   URL、独一无二的完整标题、精确到日的上传日期等）替换为自然的间接描述；
2. **保持叙事流畅**：不要为了模糊而把句子改得拗口、绕口或啰嗦；
3. **保留风格**："某频道""某期节目""近几年"这种自然语句优先，禁止输出类似"20 世纪 90
   年代中期在美国举办的夏季奥运会"这种过度包装的表达（这种改写反而会让题目变怪）；
4. **顺手删掉"视频平台"套话**。如果原 question 以"海外知名视频平台上""在 YouTube 上"
   "On a major global video-sharing platform""On YouTube"等开头或中间出现这类短语，
   直接**删除**它们，让句子自然起笔（例如"海外知名视频平台上有一个发布过 X 的频道…" →
   "有一个发布过 X 的频道…"）。这类套话对找视频没帮助，反而拖慢节奏；
5. **不要改变 question 的问点、问法和唯一性**；
6. **绝对不要**把原本"正向、唯一"的问法改成"缺失型/否定型"提问；
7. **不要**改变 answer，**不要**引入事实错误；
8. 如果原 question 已经自然又没有一击命中的致命线索，且没有上述平台套话，**直接原样输出**即可。

原始 task:
- question: {question}
- answer: {answer}
- frame_evidence: {frame_evidence}
- 其他约束：{language_instruction}

## 输出
<refined_question>
[润色或保持原样后的问题。如果无需修改，这里就是原 question。]
</refined_question>
<answer>
[与原 answer 完全一致；如发现原 answer 明显错误可小幅修正并在这里写明]
</answer>
"""


# ── Stage 3d: LLM self-check for too-easy ────────────────────────────

SELF_CHECK_SYSTEM = (
    "You are a helpful assistant. Try to answer the user's question using ONLY your "
    "internal knowledge (no tools, no guessing with hedging). If you do not know the "
    "answer with confidence, reply exactly: I don't know."
)

SELF_CHECK_CORRECTNESS_PROMPT = """Compare a model's answer against the gold answer to decide if they are semantically equivalent.

Question: {question}
Gold answer: {gold}
Model answer: {pred}

Respond with only "Yes" or "No".
"""


# ── Stage 4: search-agent verify + rewrite loop ─────────────────────
#
# Verify stage: read the text-only search agent's trace and its predicted
# answer, then decide which of the following applies:
#   * pass_task_is_hard        — the agent failed (or returned
#                                UNANSWERABLE_WITHOUT_VIDEO): the task is sound;
#   * reject_wrong_answer      — the gold answer is clearly wrong, or the search
#                                agent's results contradict it;
#   * reject_junk_entity       — the seed itself is unreasonable (a fabricated
#                                initial seed, or graph videos that are
#                                semantically misaligned with the question);
#   * too_easy_rewrite         — the agent already reached the gold answer by
#                                text search alone, so the task is too easy and
#                                the question must be rewritten using
#                                over_specific_clues.

STAGE4_VERIFY_SYSTEM = (
    "You are a strict task-quality auditor for a Video DeepResearch benchmark. "
    "You review a (question, gold_answer) pair together with the trace of a text-only "
    "search agent that just attempted to answer the question. Your job is to classify "
    "the pair and, when needed, flag over-specific clues that made the text-only agent "
    "solve it without ever watching a video."
)

STAGE4_VERIFY_USER = """## Task under review
Question:
{question}

Gold answer:
{answer}

Frame evidence (must come from watching a video frame):
{frame_evidence}

## Text-only agent attempt
Predicted answer: {agent_answer}

Abridged trace (search queries, visited URLs, critical snippets):
{trace}

## Decide
Pick exactly one verdict:
- pass_task_is_hard        : the text-only agent genuinely could NOT recover the
                              gold answer. The task is video-dependent — accept.
- too_easy_rewrite         : the agent's predicted answer matches the gold answer
                              using text-only search. List the clues in the question
                              that were too revealing so we can rewrite them.
- reject_wrong_answer      : evidence strongly suggests the gold answer is factually
                              incorrect (agent found a different well-supported answer
                              that contradicts it).
- reject_junk_entity       : the question / seed anchor is incoherent — e.g. a
                              fabricated entity, or the linked video is unrelated
                              to the premise. Unfixable by rewriting.

If and only if verdict == too_easy_rewrite, also enumerate 1-5 over_specific_clues
(short strings, copied or paraphrased from the question) that a search engine can
exploit to one-shot the answer.

Output FORMAT (strict):
<verdict>pass_task_is_hard | too_easy_rewrite | reject_wrong_answer | reject_junk_entity</verdict>
<reason>one concise sentence</reason>
<over_specific_clues>
- clue 1
- clue 2
</over_specific_clues>

If no clues apply, leave the <over_specific_clues> block empty.
"""


STAGE4_REWRITE_SYSTEM = (
    "You rewrite a question so that it is no longer solvable by text-only web search. "
    "You keep the answer exactly the same. You keep the narrative natural and the "
    "uniqueness constraints intact. You do NOT introduce factual errors."
)

STAGE4_REWRITE_USER = """## Original question (too easy for text-only search)
{question}

## Gold answer (MUST be preserved exactly)
{answer}

## Frame evidence (this is what the question SHOULD force the solver to actually watch)
{frame_evidence}

## Over-specific clues flagged by the auditor — replace or soften these
{clues}

## Optional: background properties you can use to re-anchor the question
{properties}

## Rewrite rules
1. Replace each over-specific clue with an **indirect, paraphrased description**
   that requires multi-hop reasoning to disambiguate. Typical replacements:
   - exact date → "a couple of years ago" / "近几年";
   - unique full title → "a video focused on X";
   - proper account/channel name → "a channel known for X";
   - single distinguishing property → combine 2-3 weaker properties from
     the optional background block above so the target is still unique but
     no single google query hits it.
2. Keep the question in the SAME language as the original.
3. Preserve the **uniqueness** of the answer — do NOT switch to negative /
   exhaustive phrasing (e.g. "which item is missing").
4. Do NOT mention video platforms like "YouTube" or "video-sharing platform".
5. Output only the rewritten question, wrapped in <refined_question>...</refined_question>.

<refined_question>
[your rewritten question]
</refined_question>
"""

