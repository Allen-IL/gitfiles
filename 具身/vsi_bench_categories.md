# VSI-bench VQA
## 1. 整体概览

* **总问题数：** 5,155 个
* **总场景数：** 236 个 (包含 `arkitscenes`, `scannetpp`, `scannet` 三种来源的独立场景)

## 2. 数据集来源分布
VQA 问题的数据源来自三个不同的 3D 场景数据集：

* **Scannet:** 2,209 个问题 (约 42.9%)
* **ARKitScenes:** 1,500 个问题 (约 29.1%)
* **Scannetpp:** 1,446 个问题 (约 28.0%)

## 3. 任务类型（Question Type）分布
数据集中包含 10 个不同的任务类别，具体数量分布如下：

| 任务类别 (Question Type) | 问题数量 | 占比 |
| :--- | :--- | :--- |
| `obj_appearance_order` (物体出现顺序) | 1,051 | 20.4% |
| `object_rel_distance` (物体相对距离) | 721 | 14.0% |
| `object_counting` (物体计数) | 550 | 10.7% |
| `object_size_estimation` (物体尺寸估计) | 562 | 10.9% |
| `room_size_estimation` (房间面积估计) | 250 | 4.8% |
| `object_abs_distance` (物体绝对距离) | 441 | 8.6% |
| `route_planning` (路线规划) | 495 | 9.6% |
| `object_rel_direction_hard` (相对方向-困难) | 370 | 7.2% |
| `object_rel_direction_medium` (相对方向-中等) | 382 | 7.4% |
| `object_rel_direction_easy` (相对方向-简单) | 333 | 6.5% |
| **总计** | **5,155** | **100.0%** |

## 4. 对象种类分析报告
根据 `output.json` 文件中所有VQA问答对的统计分析，VSI-bench的VQA任务共涉及 **4** 大类对象。

详细分类及包含的具体对象（共计 61 种独立对象）如下：

---

### 1. 家具 (Furniture)
*共 15 种*

* `bed` (床)
* `bookshelf` (书架)
* `chair` (椅子)
* `closet` (衣柜)
* `coat rack` (衣帽架)
* `counter` (柜台/台面)
* `cushion` (坐垫)
* `mattress` (床垫)
* `nightstand` (床头柜)
* `piano` (钢琴)
* `shoe rack` (鞋架)
* `sofa` (沙发)
* `stool` (凳子)
* `table` (桌子)
* `whiteboard` (白板)

### 2. 家电 / 电子产品 (Appliances / Electronics)
*共 22 种*

* `ceiling light` (顶灯)
* `clock` (时钟)
* `computer mouse` (鼠标)
* `computer tower` (电脑主机)
* `dishwasher` (洗碗机)
* `exhaust fan` (排风扇)
* `fan` (风扇)
* `heater` (加热器)
* `headphones` (耳机)
* `kettle` (水壶)
* `keyboard` (键盘)
* `laptop` (笔记本电脑)
* `microwave` (微波炉)
* `monitor` (显示器)
* `oven` (烤箱)
* `power strip` (插线板)
* `printer` (打印机)
* `refrigerator` (冰箱)
* `stove` (炉灶)
* `table lamp` (台灯)
* `telephone` (电话)
* `tv` (电视)
* `washer` (洗衣机)

### 3. 建筑 / 卫浴固定装置 (Structural / Fixtures)
*共 7 种*

* `bathtub` (浴缸)
* `door` (门)
* `fireplace` (壁炉)
* `mirror` (镜子)
* `radiator` (散热器)
* `toilet` (马桶)
* `window` (窗户)

### 4. 物品 / 杂物 / 装饰品 (Props / Miscellaneous / Decor)
*共 17 种 (此类也涵盖了“展品”)*

* `backpack` (背包)
* `basket` (篮子)
* `blanket` (毯子)
* `bowl` (碗)
* `bucket` (水桶)
* `crate` (板条箱)
* `cup` (杯子)
* `cutting board` (砧板)
* `guitar` (吉他)
* `pan` (平底锅)
* `paper bag` (纸袋)
* `pillow` (枕头)
* `plant` (植物)
* `shoes` (鞋子)
* `suitcase` (行李箱)
* `towel` (毛巾)
* `trash can` (垃圾桶)