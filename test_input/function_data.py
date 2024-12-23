data = [{'功能': '随速助力', '功能描述': '随速助力功能根据驾驶员输入的方向盘扭矩大小和车速，计算目标助力电流，控制电机运转以实现基本助力。具体工作机制如下：\n\n- **使能条件**：车辆模式处于任意模式（VehMod == Normal || Transport || Dyno || Crash），电源模式为Driving（UsrMod == Driving），且EHPS系统无故障。\n- **触发条件**：当驾驶员转动方向盘时（方向盘扭矩 != 0）。\n- **执行输出**：接收方向盘扭矩、驾驶模式DrvMod、车速DirVehSpd，并查表引入标定值，对应输出±随速助力扭矩。车速在0～110km/h范围内，至少保留12个以上可用于标定的细分车速曲线。可响应扭矩范围[-6Nm, +6Nm]，可识别扭矩范围[-10Nm, +10Nm]。当车速信号丢失时，EPS执行默认安全速助力，具体车速值可标定。\n- **退出条件**：无明确退出条件。\n- **退出状态**：无明确退出状态。', '相关参考内容': '1.1.1 随速助力需求描述中的所有内容', '其他补充': '该功能确保了不同车速下的转向助力效果，提高了驾驶舒适性和安全性。'}, {'功能': '主动回正', '功能描述': '主动回正功能通过方向盘转角信号主动拉方向盘回中心，提高方向盘返回功能，并且方向盘回正速度可控。具体工作机制如下：\n\n- **使能条件**：车辆模式处于任意模式（VehMod == Normal || Transport || Dyno || Crash），电源模式为Driving（UsrMod == Driving），且EHPS系统无故障。\n- **触发条件**：当驾驶员转动方向盘时（方向盘转角 != 0）。\n- **执行输出**：接收车速DirVehSpd、驾驶模式DrvMod查表得出主动回正系数，接收方向盘转角、方向盘扭矩、方向盘转角速率查表得出相应系数，引入标定值，结合限幅，输出主动回正扭矩。回正扭矩应随方向盘手力矩增大而减小；回正扭矩应随着车速增加而平稳降低，车速为0时，回正力矩为0；在主动回正的过程中，方向盘转速应可控。目标方向盘转速应随方向盘转角位置的减小而逐渐减小；当方向盘转角位置逐渐靠近中位时，目标方向盘转速应迅速减小至0。\n- **退出条件**：电源模式不为Driving（UsrMod != Driving）或EHPG识别到转角故障。\n- **退出状态**：退出主动回正控制。', '相关参考内容': '1.1.2 主动回正需求描述中的所有内容', '其他补充': '该功能增强了方向盘的自动回正能力，提升了驾驶体验和安全性。'}, {'功能': '软止点保护', '功能描述': '软止点保护功能在驾驶员将方向盘转到极限位置时，降低驱动电流输出，减少助力，保护电机与机械结构。具体工作机制如下：\n\n- **使能条件**：车辆模式处于任意模式（VehMod == Normal || Transport || Dyno || Crash），电源模式为Driving（UsrMod == Driving），且EHPS系统无故障。\n- **触发条件**：当驾驶员转动方向盘在转角行程末端前60°范围内（abs（方向盘转角）>TBD）。\n- **执行输出**：接收方向盘转角和车速DirVehSpd查表末端保护限制范围和末端保护增益补偿量，经过末端保护等级计算，输出末端保护增益。本功能需要支持通过标定方法改变末端力矩的变化梯度。\n- **退出条件**：电源模式不为Driving（UsrMod != Driving）。\n- **退出状态**：退出软止点保护。', '相关参考内容': '1.1.3 软止点保护需求描述中的所有内容', '其他补充': '该功能防止了因过度转向导致的机械损伤，延长了系统的使用寿命并提高了安全性。'}]
