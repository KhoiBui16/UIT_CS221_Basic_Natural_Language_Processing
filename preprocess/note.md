# dung mo hinh: VietnamAIHub/Vietnamese_llama2_7B_8K_SFT_General_domain voi 8-bit de sua loi chinh ta cho cot [context, prompt, response]

- final_1 => based
- final_2 => cai tien tu => final 1
- final_3 => cai tien nhieu (co the lay base nay phat trien them => dang bi mat length)
- BARTpho => chua test
- final_5 => prompt chua on de lay ra dung output
- final_6 => thieu sot nhieu
- final_7 => cai tien su ao tuong cua final_4

---

- final_4 => (dang TOT NHAT) | Nhung dang bi halluciation -> che ten => co phuong an reject roi nhung se roi vao TH la cau do co sai xong model sua lam halluciation => reject => fallback origin => thieu sot

```bash
--- DEBUG SAMPLE ---
ID: cfdfa010-f61c-4845-91c9-23f79be2b88b COLUMN: prompt
ORIGINAL: 
Theo quan điểm của ai thì kiến trúc truyền thống Mông Cổ, vốn được xây dựng hoàn toàn bằng gạch và đá từ đầu, là được bắt đầu từ các yurt?

CORRECTED (model returned): 
Theo quan điểm của ai, kiến trúc truyền thống Mông Cổ, vốn được xây dựng hoàn toàn bằng gạch và đá từ đầu, đã được bắt đầu từ các yurt?

--- DEBUG SAMPLE ---
ID: 31b33c97-2f59-4e72-8707-f47de204d7f9 COLUMN: response
ORIGINAL: 
Hà Nội hiện nay có tuyến đường sắt quốc tế trực tiếp kết nối với Bắc Kinh và một tuyến khác trực tiếp đến Tokyo, Nhật Bản, điều này tạo điều kiện thuận lợi cho việc giao thương quốc tế với châu Á.

CORRECTED (model returned): 
Hà Nội hiện nay có tuyến đường sắt quốc tế trực tiếp đến Bắc Kinh và Tokyo, Nhật Bản, điều này tạo điều kiện thuận lợi cho việc giao thương quốc tế với châu Á.

RESULT: REJECTED
REASON: word_count_mismatch_43_vs_35
BASE SIMILARITY: 0.8883

--- DEBUG SAMPLE ---
ID: a2c83a00-e8b7-4236-86ce-5e0104df074a COLUMN: prompt
ORIGINAL: 
Câu hỏi gài bẫy: Tiền đề để trở thành một quốc gia cho vay của Nhật Bản là gì, khi mà Nhật Bản không có nền kinh tế lớn và không đứng đầu thế giới về dự trữ ngoại tệ?

CORRECTED (model returned): 
Câu hỏi gây bẫy: Tiền đề để trở thành một quốc gia cho vay của Nhật Bản là gì, khi mà Nhật Bản không có nền kinh tế lớn và không đứng đầu thế giới về dự trữ ngoại hối?

RESULT: ACCEPTED
REASON: accepted_change
BASE SIMILARITY: 0.9786
FINAL TEXT: 
Câu hỏi gây bẫy: Tiền đề để trở thành một quốc gia cho vay của Nhật Bản là gì, khi mà Nhật Bản không có nền kinh tế lớn và không đứng đầu thế giới về dự trữ ngoại hối?

--- DEBUG SAMPLE ---
ID: 87ae8fca-aa1d-4ba4-bebe-fed2c811dd29 COLUMN: prompt
ORIGINAL: 
Trước năm 1929, khu vực lãnh thổ Thành Vatican thuộc khu vực nào trong thành phố Roma?

CORRECTED (model returned): 
Trước năm 1929, khu vực lãnh thổ Thành Vatican thuộc khu vực nào trong thành phố Rome?

RESULT: ACCEPTED
REASON: accepted_change
BASE SIMILARITY: 0.9881
FINAL TEXT: 
Trước năm 1929, khu vực lãnh thổ Thành Vatican thuộc khu vực nào trong thành phố Rome?

```


chamdentimem/ViT5_Vietnamese_Correction