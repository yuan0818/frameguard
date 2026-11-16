# Image/Video Quality Classifier
#
# 支持 10 类分类：
#   0 - normal                 正常
#   1 - screen_corruption      花屏
#   2 - black_screen           黑屏
#   3 - green_screen           绿屏
#   4 - blocking_artifact      块效应
#   5 - mosaic                 马赛克
#   6 - color_space_error      色彩空间出错
#   7 - ui_layout_misalignment UI 布局错位
#   8 - ui_element_overlap     UI 元素重叠
#   9 - ui_text_overlap        UI 文本重叠
#
# UI 布局检测模块 (ui_detector) 提供基于 CV 的结构化分析，
# 可与 CNN 分类结果结合，提供详细的异常定位报告。
# 适用于在 CNN 初判为 UI 异常后，进一步分析具体异常位置。
