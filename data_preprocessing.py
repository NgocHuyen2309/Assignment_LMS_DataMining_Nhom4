import pandas as pd
import logging
from pathlib import Path
from typing import List

# Cấu hình logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)

class GroceriesDataProcessor:
    """
    Lớp xử lý và chuẩn bị dữ liệu Groceries cho thuật toán Apriori.
    """
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.df: pd.DataFrame = pd.DataFrame()
        self.transactions: List[List[str]] = []

    def load_data(self) -> None:
        """Đọc dữ liệu thô từ file CSV."""
        if not self.file_path.exists():
            logging.error(f"Không tìm thấy file: {self.file_path}")
            raise FileNotFoundError(f"Vui lòng đặt file {self.file_path.name} cùng thư mục với script.")
        
        logging.info(f"Đang đọc dữ liệu từ {self.file_path.name}...")
        self.df = pd.read_csv(self.file_path)
        logging.info(f"Số lượng bản ghi gốc: {len(self.df)}")

    def clean_data(self) -> None:
        """
        Làm sạch dữ liệu (Data Cleaning):
        - Loại bỏ các dòng có giá trị rỗng (NaN).
        - Chuẩn hóa chuỗi (chữ thường, xóa khoảng trắng thừa) để tránh trùng lặp giả.
        """
        logging.info("Đang tiến hành làm sạch và chuẩn hóa dữ liệu...")
        self.df.dropna(subset=['Member_number', 'Date', 'itemDescription'], inplace=True)
        self.df['itemDescription'] = self.df['itemDescription'].astype(str).str.strip().str.lower()

    def transform_to_transactions(self) -> None:
        """
        Chuyển đổi dữ liệu bảng thành tập các giao dịch (Data Transformation).
        Sử dụng cấu trúc 'set' trong lambda để đảm bảo mỗi giỏ hàng là một Itemset độc lập,
        triệt tiêu các mặt hàng mua trùng lặp trong cùng 1 thời điểm.
        """
        logging.info("Đang cấu trúc hóa dữ liệu thành các giao dịch (Transactions)...")
        grouped = self.df.groupby(['Member_number', 'Date'])['itemDescription'].apply(lambda x: list(set(x)))
        self.transactions = grouped.tolist()
        
        # Trích xuất thống kê mô tả phục vụ báo cáo
        total_transactions = len(self.transactions)
        unique_items = len(set(item for t in self.transactions for item in t))
        avg_len = sum(len(t) for t in self.transactions) / total_transactions
        max_len = max(len(t) for t in self.transactions)
        
        logging.info("================ THỐNG KÊ DATASET ================")
        logging.info(f"Tổng số giao dịch (|D|): {total_transactions}")
        logging.info(f"Số lượng mặt hàng duy nhất: {unique_items}")
        logging.info(f"Kích thước giỏ hàng trung bình: {avg_len:.2f} items/giao dịch")
        logging.info(f"Kích thước giỏ hàng lớn nhất: {max_len} items")
        logging.info("==================================================")

    def get_transactions(self) -> List[List[str]]:
        """Trả về danh sách giao dịch cuối cùng để nạp vào thuật toán."""
        return self.transactions
    
    def export_to_csv(self, output_name: str = "cleaned_transactions.csv") -> None:
        """
        Xuất dữ liệu đã xử lý ra file CSV vật lý.
        Không gọi hàm này trong quá trình đo lường hiệu năng (Benchmark) để tránh nhiễu I/O.
        """
        import csv
        output_path = self.file_path.parent / output_name
        logging.info(f"Đang xuất dữ liệu sạch ra file minh chứng: {output_path.name}...")
        
        with open(output_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(self.transactions)
            
        logging.info("Đã lưu file minh chứng thành công! (Dùng để nộp kèm báo cáo)")

def main():
    # Sử dụng Pathlib để lấy đường dẫn tương đối, code tự nhận diện thư mục hiện tại
    current_dir = Path(__file__).parent
    dataset_file = current_dir / "Groceries_dataset.csv"
    
    processor = GroceriesDataProcessor(dataset_file)
    
    try:
        processor.load_data()
        processor.clean_data()
        processor.transform_to_transactions()
        processor.export_to_csv()  # [BẬT LÊN 1 LẦN DUY NHẤT] Xuất file clean data, sau đó comment dòng này lại
        
        # Biến transactions này sẽ được dùng để pass vào thuật toán của Thành viên 4
        transactions = processor.get_transactions()
        
    except Exception as e:
        logging.error(f"Lỗi trong quá trình tiền xử lý: {e}")

if __name__ == "__main__":
    main()