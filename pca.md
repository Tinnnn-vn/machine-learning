## Principal Component Analysis (PCA) – Thu gọn dữ liệu mà vẫn giữ phần quan trọng
> Tài liệu nhập môn Principal Component Analysis (PCA) dành cho người mới học Machine Learning.
> 
> Mục tiêu: hiểu PCA bằng trực giác trước, sau đó mới nhìn vào công thức và mã Python.

## 1. PCA giải quyết vấn đề gì?

Giả sử mỗi học sinh được mô tả bằng 10 thông tin: số giờ học, điểm Toán, điểm Văn, thời gian ngủ, số buổi nghỉ… Mỗi thông tin là một đặc trưng (feature), nên mỗi học sinh là một điểm trong không gian 10 chiều.

Không gian 10 chiều rất khó quan sát. PCA giúp biến 10 đặc trưng ban đầu thành vài đặc trưng mới nhưng vẫn giữ lại phần lớn sự khác biệt giữa các học sinh.

Ví dụ:

```
10 đặc trưng ban đầu ──PCA──> 2 thành phần chính
        khó nhìn                  dễ vẽ và phân tích
```

PCA thường được dùng để:

- nén dữ liệu và giảm số cột;
- vẽ dữ liệu nhiều chiều trên mặt phẳng;
- giảm nhiễu;
- phát hiện nhóm hoặc điểm bất thường;
- giúp một số mô hình học máy chạy nhanh hơn.

## 2. Trực giác: Tìm góc nhìn tốt nhất

Hãy tưởng tượng các điểm dữ liệu là một đám mây dài và nghiêng. Nếu chiếu chúng xuống một đường thẳng phù hợp, các “bóng” vẫn cách xa nhau và ta còn nhận ra hình dạng chính của đám mây.

- **PC1** là hướng mà các bóng trải rộng nhất. Nó giữ nhiều thông tin nhất.
- **PC2** vuông góc với PC1 và giữ phần biến đổi quan trọng tiếp theo.
- PC3, PC4… cũng được tìm tương tự.

```mermaid
flowchart LR
    A["Dữ liệu nhiều đặc trưng"] --> B["Tìm hướng biến đổi lớn nhất"]
    B --> C["Chiếu dữ liệu lên các hướng mới"]
    C --> D["Giữ vài thành phần quan trọng"]
```

Trong PCA, **thông tin** được đo gần đúng bằng **phương sai**: dữ liệu càng trải rộng trên một hướng thì hướng ấy càng giúp phân biệt các mẫu.

> PCA không chọn vài cột cũ để giữ lại. Nó tạo ra các cột mới bằng cách trộn những cột ban đầu.
