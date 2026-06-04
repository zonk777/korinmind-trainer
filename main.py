# ============================================================================
# KorinMind — 从零制作的轻量级大语言模型
# ============================================================================
# 用法：
#   python main.py                                      # 默认参数训练
#   python main.py --epochs 5 --batch_size 16            # 自定义参数
#   python trainer/trainer_pretrain.py --epochs 1 ...    # 直接运行训练脚本也可以
# ============================================================================

import os
import sys
import runpy


def main():
    # 确保工作目录是项目根目录
    project_root = os.path.dirname(os.path.abspath(__file__))
    os.chdir(project_root)
    sys.path.insert(0, project_root)

    print("=" * 60)
    print("  KorinMind — 从零训练大语言模型")
    print("=" * 60)
    print()

    # 把 main.py 收到的参数传给训练脚本
    sys.argv = ["trainer/trainer_pretrain.py"] + sys.argv[1:]

    # 在当前进程中执行训练脚本（不用 subprocess，避免终端 IO 问题）
    runpy.run_path(
        os.path.join(project_root, "trainer", "trainer_pretrain.py"),
        run_name="__main__",
    )


if __name__ == "__main__":
    main()
