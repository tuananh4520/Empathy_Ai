# Empathy AI Flexible Demo

Bản demo bám theo mô hình: hệ thống có thể chạy với từng nguồn riêng lẻ hoặc với bất kỳ tổ hợp nguồn nào do người dùng lựa chọn.

Các chế độ hỗ trợ:
- Văn bản
- Khuôn mặt
- Âm thanh
- Văn bản + Khuôn mặt
- Văn bản + Âm thanh
- Khuôn mặt + Âm thanh
- Văn bản + Khuôn mặt + Âm thanh

## 1) start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python app.py
```

Mở trình duyệt tại:

```text
http://127.0.0.1:7862
```

## 2) Cách dùng

### Demo nhanh văn bản
 - Chọn `Văn bản`
- Nhập câu ví dụ: `Tôi đang khá áp lực vì còn nhiều phần luận văn chưa xong.`
- Bấm `Phân tích và phản hồi`

### Demo trực quan
- Chọn `Văn bản` + `Khuôn mặt`
- Bật webcam
- Nhập nội dung văn bản
- Bấm `Phân tích và phản hồi`

### Demo đa phương thức đầy đủ
- Chọn `Văn bản` + `Khuôn mặt` + `Âm thanh`
- Bật webcam hoặc tải ảnh mặt
- Ghi âm 3–8 giây bằng giọng tự nhiên
- Bấm `Phân tích và phản hồi`

## 3) Thành phần hệ thống

- Nhánh văn bản: nhận diện cảm xúc từ câu người dùng nhập
- Nhánh khuôn mặt: nhận diện cảm xúc từ biểu cảm khuôn mặt
- Nhánh âm thanh: suy ra cảm xúc từ cường độ, cao độ, nhịp điệu giọng nói
- Khối hợp nhất linh hoạt: chỉ hợp nhất các nguồn mà người dùng chọn
- Khối sinh phản hồi đồng cảm: tạo phản hồi theo cảm xúc cuối và ngữ cảnh hội thoại
- Khối chấm điểm thấu cảm: tính emotion alignment, warmth, relevance

## 4) Tệp sơ đồ kiến trúc

- `architecture.md`: sơ đồ dạng Mermaid và mô tả logic hệ thống
- `architecture_flexible.png`: sơ đồ kiến trúc hệ thống dạng ảnh

## 5) Lưu ý học thuật

Đây là bản demo chạy local, tối ưu để trình bày luận văn và minh họa kiến trúc. Các nhánh hiện dùng luật/đặc trưng nhẹ để chạy trên máy cá nhân. Khi nâng lên bản nghiên cứu sâu hơn, bạn có thể thay thế:
- Văn bản -> PhoBERT/BERT tiếng Việt
- Khuôn mặt -> FER/DeepFace/CNN
- Âm thanh -> wav2vec2, CNN-BiLSTM hoặc SER model chuyên dụng
