# Phân công công việc — Store Sales Forecasting

Nhóm có **4 người**: **TV1: dữ liệu/EDA/features; TV2: cả XGBoost và LightGBM (non-DL); TV3: GRU (DL); TV4: đánh giá và XAI**. Mục tiêu chung là dự báo lượng bán của từng cửa hàng–nhóm hàng trong 16 ngày tiếp theo.

## 1. TV1 — Dữ liệu, EDA và features dùng chung

**Mục tiêu:** bàn giao dữ liệu có thể sử dụng để xây mô hình, cùng thông tin giải thích dữ liệu đó. TV2 nhận dữ liệu dạng bảng; TV3 nhận bảng nền để tổ chức thành chuỗi.

| Nhiệm vụ | Làm gì và vì sao cần làm? |
|---|---|
| Kiểm tra và thống nhất dữ liệu | Kiểm tra đủ 7 CSV, cột bắt buộc, ID, dòng trùng, giá trị thiếu và ngày bị mất. Phân biệt sales bằng 0 với chưa có dữ liệu; giữ số thập phân và nhãn nguồn. Mục đích là tránh để các model học từ dữ liệu sai hoặc mỗi người dùng một bản khác nhau. |
| Hoàn thiện bảng nền | Ghép thông tin cửa hàng, khuyến mãi, lịch và ngày lễ đúng phạm vi địa phương; bảo đảm không tăng số dòng hoặc thay đổi ID/sales sau ghép. Đây là nguồn đầu vào chung của TV2 và TV3. |
| Phân tích khám phá dữ liệu (EDA) | Dùng thống kê và biểu đồ để mô tả xu hướng, khác biệt theo thứ/cửa hàng/nhóm hàng, số 0 và khuyến mãi. Ghi rõ phát hiện và giới hạn, giúp người xây model hiểu dữ liệu trước khi chọn đầu vào. |
| Bổ sung features lịch sử dùng chung | Feature là cột thông tin đưa vào để dự đoán sales. Bổ sung sales tại mốc dự báo và trước mốc 7/14/28 ngày; trung bình, độ lệch chuẩn 7/14/28 ngày; tỷ lệ sales bằng 0 trong 28 ngày. Các cột này mô tả gần đây từng cửa hàng–nhóm hàng bán nhiều, ít hay biến động thế nào. |
| Kiểm tra và bàn giao features | Tính riêng từng cửa hàng–nhóm hàng, chỉ dùng dữ liệu đã biết tới mốc TV4 quy định; giữ cờ ngày thiếu. Bàn giao tên cột, ý nghĩa và thời điểm được phép sử dụng để TV2/TV3 không phải đoán hoặc tự ghép lại dữ liệu. |

**Sản phẩm TV1 bàn giao:** bảng nền; bảng features lịch sử cho TV2; notebook/báo cáo EDA; báo cáo chất lượng; mô tả cột và mã tái tạo dữ liệu. CSV/Parquet sinh cục bộ, không đưa lên Git.

**Giới hạn trách nhiệm:** TV1 chịu trách nhiệm dữ liệu và cách tính features chung. TV2/TV3 chịu trách nhiệm chọn cột cuối cùng, encoding/scaler riêng, huấn luyện và điều chỉnh model. TV4 chịu trách nhiệm bộ đánh giá, tổng hợp điểm và XAI. Bàn giao baseline/hàm chấm cho TV4 tiếp quản. Chi tiết đầu ra nằm trong [tài liệu features](features.md).

## 2. TV2 — Một người phụ trách cả XGBoost và LightGBM

**Mục tiêu:** xây dựng hai mô hình non-DL từ dữ liệu dạng bảng. Mỗi dòng gồm các thông tin đầu vào như cửa hàng, lịch, khuyến mãi, lịch sử bán hàng; đáp án cần học là sales.

| Nhiệm vụ | Làm gì và vì sao cần làm? |
|---|---|
| Chuẩn bị đầu vào hai model | Nhận bảng features từ TV1; xác định cột đầu vào và cột đáp án, xử lý giá trị thiếu và biến phân loại theo model. Các cột quản lý như ID được giữ để ghép kết quả, không mặc nhiên coi toàn bộ cột trong Parquet là đầu vào. |
| Xây bản đầu của cả hai model | Huấn luyện XGBoost và LightGBM trên cùng bộ features ban đầu và cùng mốc dữ liệu. Việc này cho phép quan sát khác biệt giữa hai thuật toán mà chưa trộn thêm khác biệt do bộ dữ liệu. |
| Thử nghiệm và lựa chọn cấu hình | Thử các nhóm features, cách xử lý đầu vào và tham số trên khoảng điều chỉnh chung. Ghi lại đã đổi gì và điểm thay đổi ra sao để có căn cứ chọn cấu hình cho từng model. |
| Tạo dự báo và kiểm tra | Dự báo đủ 16 ngày cho cả hai model, giữ đúng ID và chỉ dùng thông tin được phép biết. Gửi kết quả cho TV4 chấm thống nhất; tự kiểm tra dự báo không thiếu, trùng, âm hoặc không hữu hạn. |
| Hỗ trợ XAI và viết báo cáo non-DL | Bàn giao hai model đã huấn luyện, bộ tiền xử lý, danh sách features và cách gọi dự đoán cho TV4. Giải thích lựa chọn mô hình, thí nghiệm và hạn chế trong phần báo cáo non-DL. |

**Sản phẩm TV2 bàn giao:** mã train/predict và model đã lưu của cả XGBoost lẫn LightGBM; bộ tiền xử lý; danh sách features; cấu hình/thí nghiệm; dự báo theo ID; thời gian chạy; phần báo cáo non-DL.

**Giới hạn trách nhiệm:** TV2 chịu trách nhiệm hai model và đầu vào riêng của chúng. Nếu cần features chung mới, thống nhất yêu cầu với TV1; TV4 là đầu mối chấm điểm tổng hợp và thực hiện SHAP, TV2 phối hợp cung cấp model/đầu vào.

## 3. TV3 — GRU, mô hình deep learning

**Mục tiêu:** dự báo từ một đoạn lịch sử liên tiếp của mỗi cửa hàng–nhóm hàng. Đầu vào GRU được tổ chức thành chuỗi nhiều ngày, vì vậy TV3 phụ trách bước chuyển từ bảng nền sang dạng chuỗi.

| Nhiệm vụ | Làm gì và vì sao cần làm? |
|---|---|
| Chuẩn bị dữ liệu dạng chuỗi | Nhận bảng nền từ TV1, sắp theo từng cửa hàng–nhóm hàng và ngày; xử lý rõ ngày thiếu, mã hóa cột phân loại và chuẩn hóa các đầu vào cần thiết. Bản đầu dùng 56 ngày lịch sử để dự báo trực tiếp 16 ngày tiếp theo. |
| Xây dựng và huấn luyện GRU | Thiết kế mạng và đầu ra 16 ngày, lựa chọn cách học và lưu model. GRU sử dụng diễn biến nhiều ngày để học mẫu biến động, thay vì chỉ nhận một dòng features độc lập. |
| Điều chỉnh cấu hình | Thử cấu hình mạng và lựa chọn đầu vào trên khoảng điều chỉnh chung. Ghi lại cấu hình, diễn biến quá trình học, thời gian chạy và kết quả để giải thích lựa chọn cuối cùng. |
| Tạo dự báo và ghép đúng ID | Chuyển đầu ra theo chuỗi trở lại đúng ngày–cửa hàng–nhóm hàng và ID. Kiểm tra đủ dòng, trả dự báo về thang sales gốc rồi bàn giao TV4. |
| Hỗ trợ XAI và viết báo cáo DL | Gửi model, bộ tiền xử lý, mô tả từng nhóm đầu vào và hàm dự đoán để TV4 khảo sát ảnh hưởng của chúng. Viết phần kiến trúc, thí nghiệm, kết quả và hạn chế GRU. |

**Sản phẩm TV3 bàn giao:** bộ tạo dữ liệu chuỗi; model GRU và mã train/predict; bộ mã hóa/chuẩn hóa; cấu hình/thí nghiệm; dự báo theo ID; thời gian chạy; phần báo cáo DL.

**Giới hạn trách nhiệm:** TV3 tự phụ trách cửa sổ chuỗi và các bước tiền xử lý riêng của GRU. TV1 bàn giao bảng nền, không chịu trách nhiệm thiết kế mạng hoặc tạo sẵn mọi đầu vào GRU. TV3 phối hợp TV4 để thực hiện giải thích mô hình.

## 4. TV4 — Đánh giá, phân tích lỗi và XAI

**Mục tiêu:** xác định model nào dự báo tốt hơn trong cùng điều kiện và giải thích các thông tin model sử dụng để đưa ra dự báo. XAI là phần giải thích kết quả mô hình, không chỉ liệt kê điểm số.

| Nhiệm vụ | Làm gì và vì sao cần làm? |
|---|---|
| Quản lý bài kiểm tra chung | Bàn giao các mốc thời gian, tập ID và quy tắc thông tin được phép dùng cho TV1/TV2/TV3. Kiểm tra mọi model đều dự báo cùng 16 ngày, tránh việc một người được biết nhiều dữ liệu hơn người khác. |
| Tiếp quản baseline và hàm chấm | Baseline là cách dự báo đơn giản để làm mốc so sánh; ở đây dùng trung bình sales cùng thứ trong 28 ngày trước. Tiếp nhận mã đã có, tính kết quả cho hai khoảng kiểm tra và duy trì hàm RMSLE ghép theo ID. |
| Chấm và so sánh ba model | Chấm dự báo XGBoost, LightGBM và GRU bằng cùng RMSLE trên sales gốc; điểm thấp hơn là tốt hơn. Lập bảng so sánh với baseline, kèm thời gian chạy, số dòng dự báo và cấu hình tương ứng. |
| Phân tích lỗi | Xác định model sai nhiều ở nhóm hàng/cửa hàng nào, ngày dự báo gần hay xa, trường hợp sales bằng 0 hoặc lượng bán lớn. Phần này giải thích điểm mạnh/yếu mà một điểm tổng hợp không thể hiện hết. |
| Giải thích XGBoost và LightGBM | Dùng SHAP để phân tích features nào ảnh hưởng nhiều và chúng đóng góp theo hướng tăng/giảm dự báo. Có nhận xét tổng thể và một số ví dụ dự báo cụ thể, phối hợp TV2 kiểm tra ý nghĩa đầu vào. |
| Giải thích GRU | Khảo sát thay đổi dự báo khi thay/che một nhóm đầu vào hoặc một đoạn lịch sử, phối hợp TV3. Mục tiêu là chỉ ra nhóm thông tin và phần lịch sử nào model nhạy với; ghi rõ giới hạn của cách giải thích. |
| Tổng hợp kết quả nhóm | Viết phần so sánh, phân tích lỗi, XAI và hạn chế; làm đầu mối ghép kết quả các model vào báo cáo chung. Giải thích mối liên hệ model học được, không kết luận đó là nguyên nhân thực tế của lượng bán. |

**Sản phẩm TV4 bàn giao:** mốc/tập ID kiểm tra; bộ chấm dùng chung; baseline theo mốc; bảng điểm ba model; phân tích lỗi; biểu đồ và nhận xét XAI; phần đánh giá/kết luận của báo cáo.

**Giới hạn trách nhiệm:** TV4 không xây thay model cho TV2/TV3 hoặc sửa thay bảng dữ liệu của TV1. Khi phát hiện vấn đề, chỉ rõ bằng kết quả kiểm tra để người phụ trách xử lý; TV2/TV3 phải cung cấp đủ model và bộ tiền xử lý để TV4 làm XAI.

## 5. Các quyết định chung để bốn người làm cùng một bài toán

### Dữ liệu và features

- Cùng 7 CSV nguồn và cùng bảng nền; bản so sánh đầu tiên chưa dùng oil/transactions.
- TV1 tính features lịch sử riêng từng cửa hàng–nhóm hàng và chốt ở ngày cuối được biết sales. Các features đó không được cập nhật bằng sales thật trong 16 ngày đang dự báo.
- TV2 bổ sung thông tin ngày đích đã biết và số ngày dự báo trước khi tạo đầu vào cho hai model bảng; TV3 phụ trách cách biểu diễn tương ứng cho GRU. Mọi đầu vào phải tuân thủ cùng giới hạn thời gian.
- Sales trong dữ liệu bàn giao giữ nguyên. Bản đầu của ba model học đáp án đã biến đổi bằng `log1p`; TV2/TV3 chịu trách nhiệm chuyển dự báo về sales gốc và chặn giá trị âm về 0 trước khi chấm.
- Bộ mã hóa, scaler và các bước học thống kê chỉ được fit trên phần dữ liệu được dùng để học ở từng mốc. Ngày thiếu không tự đổi thành sales bằng 0.

### Các mốc được chọn

Validation là khoảng dữ liệu đã có đáp án được giữ riêng để kiểm tra: model dự báo trước, sau đó mới so với sales thật. Không xóa đáp án khỏi CSV nguồn.

| Mục đích | Chỉ được biết sales tới hết ngày | Khoảng cần dự báo |
|---|---|---|
| Chọn features và điều chỉnh model | 14/07/2017 | 15/07–30/07/2017, 16 ngày |
| So sánh cuối sau khi chốt cấu hình | 30/07/2017 | 31/07–15/08/2017, 16 ngày |
| Dự báo test bằng phương pháp được chọn | 15/08/2017 | 16/08–31/08/2017, 16 ngày |

Tất cả model dự báo trọn 16 ngày từ một mốc cố định. Ví dụ ở mốc 30/07, không được dùng sales thật của ngày 31/07 để dự báo các ngày sau. Các mẫu dùng để học cũng phải có đáp án đã biết tới mốc tương ứng. Không chia ngẫu nhiên các dòng.

TV2/TV3 lựa chọn features và tham số trên khoảng điều chỉnh; chốt cấu hình trước khi xem điểm so sánh cuối. TV4 chấm đúng 28.512 ID ở mỗi khoảng kiểm tra bằng RMSLE trên sales gốc. Baseline hiện có điểm **0,531130** ở khoảng 31/07–15/08; TV4 bổ sung điểm ở khoảng điều chỉnh. Khoảng test không có sales thật để chấm nội bộ.

## 6. Thứ tự bàn giao

1. **TV4 chốt tập ID/quy tắc đánh giá; TV1 bàn giao dữ liệu và features.** TV2/TV3 xác nhận đầu vào đủ dùng, báo lại nếu thiếu thông tin.
2. **TV2 xây cả XGBoost và LightGBM; TV3 xây GRU.** Mỗi model bàn giao dự báo `id, sales_pred`, cấu hình và thời gian chạy cho TV4. Đây là dữ liệu kết quả để bàn giao cục bộ, không đưa CSV lên Git.
3. **TV2/TV3 điều chỉnh rồi chốt cấu hình.** Sau đó chạy khoảng so sánh cuối và gửi model đã lưu, bộ tiền xử lý cùng mô tả đầu vào cho TV4.
4. **TV4 chấm điểm, phân tích lỗi và XAI.** Cả nhóm dựa trên kết quả này chọn phương pháp cuối. Chủ model được chọn huấn luyện lại với sales tới 15/08 và tạo submission; TV4 kiểm tra ID/định dạng.
5. **TV1/TV2/TV3/TV4 viết các phần báo cáo tương ứng; cả nhóm hoàn thiện báo cáo và slide.** TV4 làm đầu mối tổng hợp bảng kết quả và kết luận chung.
