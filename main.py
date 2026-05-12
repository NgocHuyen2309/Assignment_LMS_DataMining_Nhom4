import logging
import time
import tracemalloc
import csv
from pathlib import Path

from data_preprocessing import GroceriesDataProcessor
from apriori_core import run_apriori

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')

def main():
    print("\n" + "═"*70)
    print("BẮT ĐẦU CHƯƠNG TRÌNH THỰC NGHIỆM DATA MINING - APRIORI")
    print("═"*70)

    # 1. TIỀN XỬ LÝ DỮ LIỆU
    current_dir = Path(__file__).parent
    dataset_file = current_dir / "Groceries_dataset.csv"
    
    processor = GroceriesDataProcessor(dataset_file)
    processor.load_data()
    processor.clean_data()
    processor.transform_to_transactions()
    
    raw_transactions = processor.get_transactions()
    transactions = [set(t) for t in raw_transactions]

    # 2. BENCHMARK VÀ CHUẨN BỊ XUẤT FILE CHO TV6
    min_sup_thresholds = [0.05, 0.02, 0.01]
    
    # Biến lưu trữ tổng hợp dữ liệu để ghi ra file CSV
    tv6_export_data = []

    print("\n" + "═"*70)
    print("BẮT ĐẦU BENCHMARK HIỆU NĂNG VÀ BỘ NHỚ THEO NGƯỠNG MIN_SUP")
    print("═"*70)

    for min_sup in min_sup_thresholds:
        print(f"\n>>> ĐANG CHẠY THUẬT TOÁN VỚI MIN_SUP = {min_sup}")
        
        tracemalloc.start()
        start_time = time.time()
        
        # Chạy thuật toán
        result = run_apriori(transactions, min_sup)
        
        end_time = time.time()
        current_ram, peak_ram = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # Bóc tách số liệu
        runtime = end_time - start_time
        peak_ram_mb = peak_ram / (1024 * 1024)
        c_levels = result['candidates_per_level']
        
        # In log ra Terminal (Bạn chụp ảnh màn hình khúc này)
        print(f"    [+] KẾT QUẢ ĐẦU RA:")
        print(f"        - Tổng ứng viên sinh ra      : {result['total_candidates']:,} tập")
        print(f"        - Chi tiết từng vòng         : {c_levels}")
        print(f"        - Thời gian chạy (Runtime)   : {runtime:.4f} giây")
        print(f"        - Đỉnh điểm ngốn RAM (Peak)  : {peak_ram_mb:.4f} MB")

        # Lưu dữ liệu vào mảng để xuất CSV
        tv6_export_data.append({
            'min_sup': min_sup,
            'Runtime (s)': round(runtime, 4),
            'Peak_RAM (MB)': round(peak_ram_mb, 4),
            'Candidate_C1': c_levels.get('C1', 0),
            'Candidate_C2': c_levels.get('C2', 0),
            'Candidate_C3': c_levels.get('C3', 0),
            'Candidate_C4': c_levels.get('C4', 0)
        })

    # 3. TỰ ĐỘNG XUẤT FILE CSV CHO THÀNH VIÊN 6
    export_file = current_dir / "benchmark_results.csv"
    with open(export_file, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['min_sup', 'Runtime (s)', 'Peak_RAM (MB)', 'Candidate_C1', 'Candidate_C2', 'Candidate_C3', 'Candidate_C4'])
        writer.writeheader()
        writer.writerows(tv6_export_data)

    print("\n" + "═"*70)
    print(f"ĐÃ TẠO XONG FILE SỐ LIỆU: {export_file.name}")
    print("═"*70 + "\n")

if __name__ == "__main__":
    main()