import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def draw_apriori_analysis(csv_path='benchmark_results.csv'):
    """
    Hàm vẽ biểu đồ phân tích hiệu năng thuật toán Apriori.
    Sử dụng dữ liệu từ benchmark_results.csv
    """
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Lỗi: Không tìm thấy file {csv_path}")
        return

    # --- BIỂU ĐỒ 2: HIỆU NĂNG (RUNTIME & RAM) ---
    # Mục tiêu: Chứng minh khi min_sup giảm, chi phí tài nguyên tăng phi mã.
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    
    color_runtime = 'tab:red'
    ax1.set_xlabel('Ngưỡng hỗ trợ tối thiểu (min_sup)')
    ax1.set_ylabel('Thời gian chạy (giây)', color=color_runtime, fontweight='bold')
    line1, = ax1.plot(df['min_sup'], df['Runtime (s)'], marker='o', linewidth=2, color=color_runtime, label='Thời gian chạy (s)')
    ax1.tick_params(axis='y', labelcolor=color_runtime)
    ax1.invert_xaxis() # Đảo trục X vì giá trị min_sup giảm dần làm độ khó tăng lên

    ax2 = ax1.twinx()  
    color_ram = 'tab:blue'
    ax2.set_ylabel('RAM Peak (MB)', color=color_ram, fontweight='bold')
    line2, = ax2.plot(df['min_sup'], df['Peak_RAM (MB)'], marker='s', linestyle='--', color=color_ram, label='Bộ nhớ RAM (MB)')
    ax2.tick_params(axis='y', labelcolor=color_ram)

    plt.title('FIG 2: The Relationship Between MIN_SUP, RUNTIME, AND RAM', pad=20, fontweight='bold')
    
    # Gộp chú thích (Legend)
    lines = [line1, line2]
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right')
    
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.tight_layout()
    plt.savefig('fig 2: performance_chart.png', dpi=300)
    print("Đã tạo xong: performance_chart.png")

    # --- BIỂU ĐỒ 1: BÙNG NỔ TỔ HỢP (CANDIDATE GENERATION) ---
    # Mục tiêu: Minh họa trực quan sự tăng vọt của các tập ứng viên C2.
    labels = [f"min_sup {s}" for s in df['min_sup']]
    x = np.arange(len(labels))
    width = 0.25

    fig2, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(x - width, df['Candidate_C1'], width, label='C1 (1-itemsets)', color='#7fb3d5')
    rects2 = ax.bar(x, df['Candidate_C2'], width, label='C2 (2-itemsets)', color='#2980b9')
    rects3 = ax.bar(x + width, df['Candidate_C3'], width, label='C3 (3-itemsets)', color='#154360')

    ax.set_ylabel('Số lượng tập Candidates', fontweight='bold')
    ax.set_title('FIG 1: Grouped Bar Chart of Candidate Sets', pad=20, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Thêm số liệu cụ thể lên đầu cột
    for rect in [rects1, rects2, rects3]:
        for r in rect:
            height = r.get_height()
            if height > 0:
                ax.annotate(f'{int(height)}',
                            xy=(r.get_x() + r.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig('fig 1: candidate_explosion_chart.png', dpi=300)
    print("Đã tạo xong: candidate_explosion_chart.png")

if __name__ == "__main__":
    draw_apriori_analysis()