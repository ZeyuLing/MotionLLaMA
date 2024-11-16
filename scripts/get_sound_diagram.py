import sys
import librosa
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors


def plot_intensity_colored_log_scale(audio_path, frame_length=4096, hop_length=512, cmap_name='viridis',
                                     save_path=None):
    """
    绘制音频的声音强弱柱状图（基于RMS能量），每根柱子的颜色不同，并使用对数 y 轴

    参数：
    - audio_path: 音频文件路径
    - frame_length: 每帧的样本数（默认4096）
    - hop_length: 帧移（每次滑动的样本数，默认512）
    - cmap_name: 颜色映射名称（默认使用 'viridis'）
    - save_path: 保存图像的路径（可选）
    """
    try:
        # 加载音频文件
        y, sr = librosa.load(audio_path, sr=None)
    except Exception as e:
        print(f"无法加载音频文件: {e}")
        sys.exit(1)

    # 计算每帧的RMS能量
    rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]

    # 为避免对数刻度中的零值，添加一个极小值
    rms = np.where(rms == 0, 1e-10, rms)

    # 生成时间轴
    frames = range(len(rms))
    times = librosa.frames_to_time(frames, sr=sr, hop_length=hop_length)

    # 归一化RMS值以映射颜色
    norm = mcolors.LogNorm(vmin=rms.min(), vmax=rms.max())
    cmap = plt.get_cmap(cmap_name)
    colors = cmap(norm(rms))

    # 创建图形和轴
    fig, ax = plt.subplots(figsize=(14, 5))

    # 绘制柱状图
    bars = ax.bar(times, rms, width=hop_length / sr, color=colors, edgecolor='none')

    # 设置标签和标题
    ax.set_xlabel("时间 (秒)", fontsize=14)
    ax.set_ylabel("RMS 能量 (对数刻度)", fontsize=14)
    ax.set_title("音频声音强弱柱状图（彩色版，对数 y 轴）", fontsize=16)

    # 设置 y 轴为对数刻度
    ax.set_yscale('log')

    # 添加网格
    ax.grid(True, which="both", linestyle='--', alpha=0.7)

    # 创建一个 ScalarMappable 并添加颜色条
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array(rms)  # 关联 RMS 数据
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label('RMS 能量 (对数刻度)', fontsize=12)

    plt.tight_layout()

    # 保存图像（如果提供了保存路径）
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"图像已保存为 {save_path}")

    plt.show()


def main():
    if len(sys.argv) < 2:
        print(
            "用法: python plot_audio_intensity_colored.py <音频文件路径> [颜色映射名称] [保存图像路径] [使用对数刻度（yes/no）]")
        print("示例颜色映射名称: viridis, plasma, inferno, magma, cividis")
        sys.exit(1)

    audio_path = sys.argv[1]
    cmap_name = sys.argv[2] if len(sys.argv) > 2 else 'viridis'
    save_path = sys.argv[3] if len(sys.argv) > 3 else None
    use_log_scale = sys.argv[4].lower() if len(sys.argv) > 4 else 'no'

    if use_log_scale in ['yes', 'y', 'true', '1']:
        plot_intensity_colored_log_scale(audio_path, cmap_name=cmap_name, save_path=save_path)
    else:
        # 调用之前的函数，您可以根据需要保留之前的设置
        print("请使用适当的函数调用方式。当前脚本仅支持对数刻度的绘图。")
        sys.exit(1)


if __name__ == "__main__":
    main()
