# Báo cáo bài thực hành Ngày 1 – Đọc prediction YOLO11 để hiểu định dạng nhãn

**Ngày chạy:** 6/11/2026

**Runtime Colab:** CPU/GPU

**Python / PyTorch / Ultralytics:**

**Checkpoint:** `yolo11n-cls.pt`, `yolo11n.pt`, `yolo11n-seg.pt`

**Ngưỡng detection/segmentation dùng khi tạo JSON:** `0.35` (chỉ đổi nếu GV cho phép; nếu đổi hãy ghi rõ)

**Thay đổi so với notebook nguồn:** Không / mô tả rõ thay đổi

> **Cách dùng mẫu này:** copy/điền trên máy, lưu thành `REPORT.md`, rồi **upload** vào panel Files
> của Colab (notebook không tự tạo file này). Ô cuối Colab sẽ đóng ZIP cùng `day1_lab_outputs/`.
> Giải nén và đưa vào `report/` trên GitHub, rồi nộp **link repo** trên LMS.
> Không ghi họ tên, MSSV, email, số điện thoại hoặc dữ liệu cá nhân khác vào báo cáo/output;
> họ tên/MSSV chỉ nằm trong tên repository.

### Từ cần nhớ (đọc nhanh trước khi viết)

| Từ | Nghĩa ngắn |
| --- | --- |
| Prediction | Kết quả model vừa đoán ra; chưa phải nhãn đúng |
| Ground truth | Nhãn do người gán theo quy tắc |
| Taxonomy | Danh sách tên lớp có sẵn kèm model (vd. ImageNet-1K, COCO-80) |
| Checkpoint | File model `.pt` đã huấn luyện sẵn |
| Guideline | Quy tắc gán nhãn do người viết |
| Annotator | Người gán nhãn |
| Reviewer / QC | Người kiểm tra nhãn |
| Escalation | Ca khó → hỏi Lab Coach, không tự bịa |
| Model score | Số “tin” của model; không phải điểm chất lượng nhãn người |

## 1. Phân loại ảnh

**Mở file:** `day1_lab_outputs/classification_predictions.json` và `visuals/classification_top5.png`.  
Chỉ dùng ảnh `sample_id = traffic`, lấy dòng `rank = 1`.

1. Chép record hạng 1: `class_id`= 468 ; `class_name`= "cab" ; `rank`=1 ; `score`= 0.510915 ; `taxonomy_name`= "ImageNet-1K"
2. Record này nói về **cả bức ảnh** hay về **một vật thể riêng** trong ảnh? Vì sao bạn nghĩ vậy?
    Record này nói về cả bức ảnh. Vì tác vụ phân loại ảnh (image classification) dự đoán một nhãn tổng thể đại diện cho nội dung chính của toàn bộ bức ảnh, thay vì khoanh vùng tọa độ của từng vật thể riêng lẻ bên trong.
3. Danh sách tên lớp (class list) do đâu mà có: model tự nghĩ ra, hay đi kèm taxonomy/checkpoint?
    Danh sách tên lớp đi kèm taxonomy/checkpoint đã được huấn luyện sẵn (trong trường hợp này là ImageNet-1K chứa 1000 lớp). Mô hình không tự nghĩ ra tên lớp.
4. Vì sao cần ghi cả `class_id`, `class_name` và `taxonomy_name`, không chỉ ghi mỗi tên lớp?
    Vì mỗi taxonomy (ví dụ: ImageNet-1K, COCO) có một hệ thống danh sách và mã ID khác nhau. Ghi rõ taxonomy_name và class_id giúp định danh chính xác tuyệt đối lớp đó thuộc tiêu chuẩn nào, tránh nhầm lẫn khi các bộ dữ liệu khác nhau có cùng class_name nhưng khác ý nghĩa hoặc khác class_id.
5. Nếu trong ảnh có nhiều vật / nhiều chủ đề, guideline cần nói rõ điều gì để người gán nhãn chọn **một** lớp?
    Guideline cần quy định rõ nguyên tắc chọn lớp ưu tiên. Ví dụ: Ưu tiên vật thể có kích thước chiếm diện tích lớn nhất, vật thể nằm ở trung tâm bức ảnh, hoặc vật thể thể hiện rõ nhất chủ đề của bối cảnh (như trong ảnh giao thông thì ưu tiên xe cộ thay vì người đi bộ hay biển báo).
6. Vì sao không được lấy `score` của model làm ground truth?
    Vì score (hay confidence) chỉ là điểm tự tin của mô hình dùng để xếp hạng dự đoán (prediction). Nó không phản ánh chất lượng hay sự thật khách quan của dữ liệu (ground truth). Ground truth phải do con người gán dựa trên guideline.

## 2. Phát hiện vật thể

**Mở file:** `day1_lab_outputs/detection_predictions.json` và `visuals/detection_predictions.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.  
Số lượng prediction theo ngưỡng: xem output ô so sánh `0.20` / `0.35` / `0.60` trong notebook.

1. Chép một record: `class_name`= = "person" ; `score`= 0.912625 ; `bbox_xyxy`=[385.33, 69.24, 498.92, 348.92] ; `bbox_width`= 113.58 ; `bbox_height`= 279.68
2. Giải thích box bằng lời thường: góc trên-trái ở đâu, góc dưới-phải ở đâu, rộng/cao bao nhiêu pixel?
    Hộp giới hạn bao quanh vật thể có góc trên cùng bên trái bắt đầu tại pixel x = 385.33, y = 69.24 và góc dưới cùng bên phải kết thúc tại pixel x = 498.92, y = 348.92. Hộp này có chiều rộng là 113.58 pixel và chiều cao là 279.68 pixel. Gốc tọa độ (0,0) nằm ở góc trên cùng bên trái của bức ảnh.
3. Chọn hai ngưỡng (vd. 0.20 và 0.60). Mỗi ngưỡng có bao nhiêu prediction?
    Tại ngưỡng 0.20 có 2 predictions; tại ngưỡng 0.60 có 2 predictions.
4. Khi hạ ngưỡng (giữ nhiều hộp hơn), người **reviewer** phải làm thêm việc gì?
    Người reviewer/QC phải làm việc vất vả hơn để kiểm tra và loại bỏ các hộp bị sai (false positives). Do ngưỡng thấp, mô hình sẽ nhạy cảm hơn và thường xuyên khoanh nhầm vào vùng nền (background) hoặc nhận diện sai các vết mờ/đồ vật không liên quan thành đối tượng mục tiêu.
5. Viết **một quy tắc box** để người gán nhãn làm theo (áp dụng được cho nhiều ảnh, không chỉ ảnh này).
    Hộp bounding box phải vẽ bám sát (khít) nhất có thể vào tất cả các điểm mép ngoài cùng của phần diện tích nhìn thấy được của đối tượng. Tuyệt đối không để thừa nhiều khoảng trống rác xung quanh vật thể và không được cắt lẹm vào rìa của vật thể.
6. Trên ảnh `kitchen`, chỉ ra **một** vật bị che khuất hoặc cắt mép. Bạn sẽ viết thành quy tắc trong guideline, hay đưa lên hỏi Lab Coach (escalation)? Vì sao?
    Trên ảnh kitchen, nhân vật người ở rìa trái màn hình (person 0.61) và chiếc lò nướng (oven 0.69) bị mép ảnh cắt ngang làm mất một phần. Chiếc bát trắng (bowl 0.46) cũng bị che khuất một góc bởi chiếc chảo treo phía trước.
    Khi xảy ra hiện tượng vật thể bị che khuất đấy tôi sẽ viết cách xử lý vào Guideline. Ví dụ: "Nếu đối tượng bị cắt bởi mép ảnh, chỉ vẽ hộp bọc quanh phần còn hiển thị trong ảnh và gắn thuộc tính 'truncated' (bị cắt mép). Nếu đối tượng bị đè lấp, bọc khít phần nhìn thấy được và gắn thuộc tính 'occluded' (bị che khuất)."
    Lý do: Tình trạng bị che khuất hoặc cắt mép xảy ra ở hầu hết các bức ảnh thực tế. Việc chuẩn hóa chúng ngay trong guideline giúp mọi annotator thống nhất cách làm việc độc lập, đảm bảo tiến độ thay vì việc gì cũng phải hỏi escalation (nhờ vả cấp trên).

## 3. Phân đoạn theo từng đối tượng

**Mở file:** `day1_lab_outputs/segmentation_predictions.json` và `visuals/segmentation_prediction.png`.  
Chỉ dùng ảnh `sample_id = kitchen`. Chọn **một** record.

1. Chép một record: `instance_id`= "kitchen-003" ; `class_name`= "bowl" ; `score`= 0.698698 ; số điểm polygon=`polygon_point_count`= 70 ; 3–5 điểm đầu của `polygon_xy`= [84.0, 289.0], [83.0, 290.0], [81.0, 290.0], [80.0, 291.0]
2. So với box, polygon cho thêm thông tin gì về hình dạng vật thể?
    So với box (chỉ khoanh được vùng chữ nhật bao ngoài), polygon (đa giác) cung cấp thông tin chi tiết về đường viền (contour) bao quanh chính xác biên dạng thực tế của chiếc bát (bowl). Điều này giúp tách biệt chính xác các pixel thuộc về chiếc bát và các pixel thuộc về mặt bàn/hậu cảnh.
3. `instance_id` dùng để làm gì? Nó **không** phải class ID và **không** phải tracking ID — bạn hiểu câu này thế nào?
    instance_id (ở đây là kitchen-003) dùng để định danh duy nhất cho một cá thể vật thể cụ thể trong bức ảnh này (phân biệt chiếc bát số 3 với chiếc bát số 1 hay số 2). Nó không phải class_id vì có nhiều chiếc bát cùng chung class "bowl". Nó cũng không phải tracking ID vì ID này không có tác dụng liên kết hay theo vết chiếc bát đó nếu nó di chuyển trong một đoạn video.
4. Viết **một quy tắc vẽ biên mask** để người gán nhãn làm theo.
    Các điểm đa giác phải bám sát (khít) theo đường viền mép ngoài thực tế của vật thể. Không được phép vẽ lẹm vào phần nền (background), và tuyệt đối không bao gồm bóng đổ (shadows) của vật thể.
5. Trên ảnh `kitchen`, chỉ ra **một** chỗ biên mờ / hai vật chạm nhau / bị che. Bạn dùng guideline hay escalation? Vì sao?
    Trên ảnh kitchen, có hiện tượng bị che khuất và chạm nhau (ví dụ chiếc bát bowl 0.46 bị cái chảo phía trước che mất một phần biên, hay bát bowl 0.72 bị cây cán bột đè lên).
    Tôi sẽ dùng Guideline. Ví dụ quy tắc: "Khi đối tượng bị che khuất bởi một vật khác, chỉ vẽ đa giác bám theo phần hiển thị thực tế nhìn thấy được của đối tượng, không cố tình nội suy hay đoán phần bị che lấp."
    Lý do: Hiện tượng vật thể đè lên nhau trong bối cảnh nhà bếp rất phổ biến. Quy định rõ trong guideline giúp annotator có tiêu chuẩn xử lý đồng nhất và xử lý nhanh chóng mà không cần tốn thời gian hỏi (escalation).

## 4. Vòng đời và kiểm tra chất lượng

Lab hôm nay chỉ **xem prediction**. Chuỗi dưới đây là vòng đời production để bạn nối các khái niệm:

`ảnh thô → guideline → ground truth → huấn luyện → prediction → QC/rework`

Điền bảng (viết ngắn, có dẫn chứng từ JSON/PNG khi nói về lỗi):

| Tác vụ | Ground truth trông như thế nào? (đơn vị nhãn) | Lỗi hoặc chỗ khó quyết định bạn thấy trên ảnh/JSON | Annotator làm gì? | Reviewer xem gì? |
| --- | --- | --- | --- | --- |
| Phân loại ảnh |1 nhãn lớp duy nhất áp dụng cho toàn bộ bức ảnh.| Ảnh có quá nhiều vật thể (cả lò nướng, bát, người) khiến khó xác định chủ đề chính. |Dựa vào guideline để chọn 1 lớp ưu tiên phù hợp nhất.  | Kiểm tra nhãn được chọn có tuân thủ đúng mức ưu tiên của guideline hay không. |
| Phát hiện vật thể |Tên lớp (class) + Tọa độ hộp hình chữ nhật (bbox) khoanh quanh vật.|Hộp vẽ cắt lẹm vào vật thể (như bị lẹm vào cái lò nướng ở rìa).|Vẽ hộp khít quanh phần nhìn thấy của vật và gán nhãn.|Đánh giá độ bám khít của hộp, không để thừa khoảng trống rác và class đã gán đúng chưa. |
| Instance segmentation |Tên lớp (class) + Chuỗi tọa độ đa giác (polygon) cho từng cá thể. |Biên của chiếc bát mờ nhòa với màu mặt bàn, hoặc bị cán bột đè lên lẹm mất một góc. |Chấm các điểm đa giác bám sát theo đường viền mép vật thể thực tế.|Đánh giá độ chuẩn xác của đa giác so với mép thực, đảm bảo không có pixel của vùng background. |

## 5. An toàn dữ liệu

1. Quy tắc bảo vệ dữ liệu của bạn cho bài lab này là gì? (viết 1 câu)
    Chỉ sử dụng các ảnh do nội bộ cung cấp, tuyệt đối không chèn thông tin cá nhân vào nội dung báo cáo hay file JSON, và không tải dữ liệu nội bộ lên các nền tảng công khai.
2. Nếu gặp ảnh hoặc dữ liệu không đúng phạm vi, bạn sẽ dừng và báo cho ai?
    Dừng ngay việc xử lý và báo cáo (escalation) ngay lập tức cho Giảng viên hoặc Lab Coach.

## 6. Checklist trước khi nộp link repo

Kiểm tra trên GitHub trong thư mục `report/`:

- [x] `REPORT.md` (file này đã điền)
- [x] `day1_lab_outputs/classification_predictions.json`
- [x] `day1_lab_outputs/detection_predictions.json`
- [x] `day1_lab_outputs/segmentation_predictions.json`
- [x] `day1_lab_outputs/IMAGE_ATTRIBUTION.md`
- [x] `day1_lab_outputs/visuals/classification_top5.png`
- [x] `day1_lab_outputs/visuals/detection_predictions.png`
- [x] `day1_lab_outputs/visuals/segmentation_prediction.png`
- [x] Ô validation cuối notebook báo `PASS`
- [x] Không có họ tên, MSSV hoặc dữ liệu nhạy cảm trong báo cáo/output
- [x] Đã nộp **link repository** trên LMS
